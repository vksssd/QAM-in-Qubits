"""
deploy.py
---------
GSQ deployment pipeline — Phases III, IV, V.

Loads trained weights and training trajectory, runs three deployment branches:
  1. Ideal      — unquantized continuous parameters (upper bound)
  2. Euclidean  — standard K-means quantization on training trajectory (baseline)
  3. GSQ        — circular manifold clustering + soft projection (proposed)

Both baselines are fitted on the SAME real training trajectory, ensuring
a fair apples-to-apples comparison (different distance metrics, same data).

Computes per-branch:
  - Task accuracy
  - Deployment shock (Delta)
  - Fubini-Study distortion (D_FS)

Saves results/deployment_results.json for plotting.

Usage:
    python deploy.py --K 8 --alpha 0.5 --save_dir results
"""

import argparse
import json
import os
import numpy as np
import torch
import pennylane as qml

from config import GSQConfig
from data import load_two_moons_deploy
from models import VQCClassifier, hardware_efficient_ansatz
from geometry import (
    CircularKMeans, quantize_params, euclidean_quantize,
    fubini_study_distortion, deployment_shock
)


# ─────────────────────────────────────────────
# State vector extraction (Issue #6: build qnode once)
# ─────────────────────────────────────────────

class StatevectorExtractor:
    """
    Extracts quantum state vectors efficiently by building the device
    and qnode ONCE, then reusing across multiple calls.
    """

    def __init__(self, n_qubits: int, n_layers: int, n_features: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        n_features = self.n_features

        @qml.qnode(self.dev, interface="numpy")
        def statevec_circuit(params, x):
            qml.AngleEmbedding(x, wires=range(min(n_features, n_qubits)))
            p = params.reshape(n_layers, n_qubits, 2)
            hardware_efficient_ansatz(p, n_qubits, n_layers)
            return qml.state()

        return statevec_circuit

    def extract(self, X: torch.Tensor, params_flat: np.ndarray) -> np.ndarray:
        """
        Extract quantum state vectors for each input sample.
        Returns: (n_samples, 2^n_qubits) complex array.
        """
        params_np = params_flat.astype(float)
        states = []
        for xi in X.numpy():
            sv = self._circuit(params_np, xi)
            states.append(np.array(sv))
        return np.array(states)


# ─────────────────────────────────────────────
# Output (expectation value) extraction
# ─────────────────────────────────────────────

def get_outputs(model: VQCClassifier, X: torch.Tensor,
                params_flat: np.ndarray) -> np.ndarray:
    """
    Get Pauli-Z expectation values for each input using given parameters.
    """
    orig = model.params.detach().clone()
    model.set_params_flat(torch.tensor(params_flat, dtype=torch.float32))
    with torch.no_grad():
        outputs = model(X).numpy()
    model.set_params_flat(orig)
    return outputs


def accuracy_from_outputs(outputs: np.ndarray, labels: np.ndarray) -> float:
    preds = np.sign(outputs)
    preds[preds == 0] = 1
    return float(np.mean(preds == labels))


# ─────────────────────────────────────────────
# Main deployment evaluation
# ─────────────────────────────────────────────

def deploy(cfg: GSQConfig = None) -> dict:
    if cfg is None:
        cfg = GSQConfig()

    weights_path = os.path.join(cfg.save_dir, "trained_weights.pt")
    assert os.path.exists(weights_path), \
        f"Weights not found at {weights_path}. Run train.py first."

    checkpoint = torch.load(weights_path, weights_only=True)
    n_qubits  = checkpoint["n_qubits"]
    n_layers  = checkpoint["n_layers"]
    params_pt = checkpoint["params"]

    # Load training trajectory for baseline fitting (Issues #1, #2)
    assert "training_trajectory" in checkpoint, (
        "Checkpoint missing 'training_trajectory'. Re-run train.py with the "
        "updated code to save parameter snapshots during training."
    )
    training_trajectory = checkpoint["training_trajectory"].numpy()

    print(f"\n{'='*55}")
    print(f"  GSQ Deployment  |  K={cfg.K}  alpha={cfg.alpha}")
    print(f"  Circuit: {n_qubits}q  {n_layers}L")
    print(f"  Training trajectory: {training_trajectory.shape[0]} snapshots")
    print(f"{'='*55}\n")

    # Rebuild model and load weights
    model = VQCClassifier(n_qubits=n_qubits, n_layers=n_layers,
                          n_features=cfg.n_features)
    model.set_params_flat(params_pt)

    theta = params_pt.numpy().copy()  # (N,) trained parameters

    # Load test data using saved scaler (Issue #3)
    X_te, y_te = load_two_moons_deploy(
        n_samples=cfg.n_samples,
        noise=cfg.noise,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        save_dir=cfg.save_dir,
    )

    # Build statevector extractor once (Issue #6)
    sv_extractor = StatevectorExtractor(n_qubits, n_layers, cfg.n_features)

    # ── Branch 1: Ideal (no quantization) ──────────────────────────
    print("  [1/3] Ideal deployment (continuous params)...")
    outputs_ideal = get_outputs(model, X_te, theta)
    states_ideal  = sv_extractor.extract(X_te, theta)
    acc_ideal     = accuracy_from_outputs(outputs_ideal, y_te)
    print(f"        Accuracy: {acc_ideal:.3f}")

    # ── Branch 2: Euclidean K-means (baseline) ─────────────────────
    # Fitted on REAL training trajectory (Issue #1 fix)
    print(f"  [2/3] Euclidean quantization (K={cfg.K})...")
    theta_euclid   = euclidean_quantize(
        theta, training_trajectory, K=cfg.K, random_state=cfg.seed
    )
    outputs_euclid = get_outputs(model, X_te, theta_euclid)
    states_euclid  = sv_extractor.extract(X_te, theta_euclid)
    acc_euclid     = accuracy_from_outputs(outputs_euclid, y_te)
    D_FS_euclid    = fubini_study_distortion(states_ideal, states_euclid)
    delta_euclid   = deployment_shock(outputs_ideal, outputs_euclid)
    print(f"        Accuracy: {acc_euclid:.3f}  "
          f"D_FS: {D_FS_euclid:.4f}  Delta: {delta_euclid:.4f}")

    # ── Branch 3: GSQ (proposed) ───────────────────────────────────
    # Fitted on REAL training trajectory (Issue #2 fix)
    print(f"  [3/3] GSQ circular quantization (K={cfg.K}, alpha={cfg.alpha})...")

    # Fit circular K-means on the real training trajectory
    kmeans = CircularKMeans(K=cfg.K, random_state=cfg.seed)
    kmeans.fit(training_trajectory)

    theta_relaxed, theta_gsq = quantize_params(
        theta, kmeans, alpha=cfg.alpha,
        n_relaxation_steps=cfg.n_relaxation_steps
    )
    outputs_gsq = get_outputs(model, X_te, theta_gsq)
    states_gsq  = sv_extractor.extract(X_te, theta_gsq)
    acc_gsq     = accuracy_from_outputs(outputs_gsq, y_te)
    D_FS_gsq    = fubini_study_distortion(states_ideal, states_gsq)
    delta_gsq   = deployment_shock(outputs_ideal, outputs_gsq)
    print(f"        Accuracy: {acc_gsq:.3f}  "
          f"D_FS: {D_FS_gsq:.4f}  Delta: {delta_gsq:.4f}")

    # ── Summary ────────────────────────────────────────────────────
    results = {
        "K": cfg.K, "alpha": cfg.alpha,
        "n_qubits": n_qubits, "n_layers": n_layers,
        "trajectory_size": int(training_trajectory.shape[0]),
        "ideal":     {"accuracy": acc_ideal,   "D_FS": 0.0,           "delta": 0.0},
        "euclidean": {"accuracy": acc_euclid,  "D_FS": D_FS_euclid,   "delta": delta_euclid},
        "gsq":       {"accuracy": acc_gsq,     "D_FS": D_FS_gsq,      "delta": delta_gsq},
        "improvement": {
            "acc_vs_euclid":   acc_gsq - acc_euclid,
            "D_FS_reduction":  (D_FS_euclid - D_FS_gsq) / (D_FS_euclid + 1e-9),
            "shock_reduction": (delta_euclid - delta_gsq) / (delta_euclid + 1e-9),
        }
    }

    out_path = os.path.join(cfg.save_dir, "deployment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  {'─'*45}")
    print(f"  {'Method':<14} {'Accuracy':>10} {'D_FS':>10} {'Δ (shock)':>10}")
    print(f"  {'─'*45}")
    print(f"  {'Ideal':<14} {acc_ideal:>10.3f} {'—':>10} {'—':>10}")
    print(f"  {'Euclidean':<14} {acc_euclid:>10.3f} {D_FS_euclid:>10.4f} {delta_euclid:>10.4f}")
    print(f"  {'GSQ (ours)':<14} {acc_gsq:>10.3f} {D_FS_gsq:>10.4f} {delta_gsq:>10.4f}")
    print(f"  {'─'*45}")
    print(f"\n  D_FS reduction:  {results['improvement']['D_FS_reduction']*100:.1f}%")
    print(f"  Shock reduction: {results['improvement']['shock_reduction']*100:.1f}%")
    print(f"\n  Saved -> {out_path}\n")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSQ Deployment — Phases III-V")
    parser.add_argument("--K",                  type=int,   default=None)
    parser.add_argument("--alpha",              type=float, default=None)
    parser.add_argument("--n_relaxation_steps", type=int,   default=None)
    parser.add_argument("--save_dir",           type=str,   default=None)
    parser.add_argument("--seed",               type=int,   default=None)
    args = parser.parse_args()

    cfg = GSQConfig.from_args(args)
    deploy(cfg)
