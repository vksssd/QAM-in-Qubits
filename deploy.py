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
from data import load_dataset_deploy
from models import VQCClassifier, hardware_efficient_ansatz
from geometry import (
    CircularKMeans, quantize_params, euclidean_quantize,
    fubini_study_distortion, deployment_shock
)


def grid_round(theta: np.ndarray, K: int) -> np.ndarray:
    """
    Direct grid rounding to K equally spaced anchors on [-pi, pi] under circular distance.
    """
    anchors = np.linspace(-np.pi, np.pi, K, endpoint=False)
    diff = theta[:, np.newaxis] - anchors[np.newaxis, :]
    wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
    dists = np.abs(wrapped)
    nearest_idx = np.argmin(dists, axis=1)
    return anchors[nearest_idx].copy()


# ─────────────────────────────────────────────
# State vector extraction (Issue #6: build qnode once)
# ─────────────────────────────────────────────

class StatevectorExtractor:
    """
    Extracts quantum state vectors (or density matrices under noise) efficiently
    by building the device and qnode ONCE, then reusing across multiple calls.
    """

    def __init__(self, n_qubits: int, n_layers: int, n_features: int = 2,
                 noise_model: str = "none", p_depol: float = 0.0, p_damping: float = 0.0,
                 backend: str = "qiskit.aer"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.noise_model = noise_model
        self.p_depol = p_depol
        self.p_damping = p_damping
        self.backend = backend

        if backend == "qiskit.aer":
            from qiskit_aer import AerSimulator
            import qiskit_aer.noise as noise
            if noise_model != "none":
                ibm_noise = noise.NoiseModel()
                err_1q = noise.depolarizing_error(p_depol, 1).compose(noise.phase_damping_error(p_damping))
                err_2q = noise.depolarizing_error(p_depol, 2)
                ibm_noise.add_all_qubit_quantum_error(err_1q, ['rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'h', 'x', 'y', 'z'])
                ibm_noise.add_all_qubit_quantum_error(err_2q, ['cx'])
                self.sim = AerSimulator(method="density_matrix", noise_model=ibm_noise)
            else:
                self.sim = AerSimulator(method="statevector")
            
            # Precompute bit-reversal index map for fast big-endian Pennylane alignment
            self.idx_map = [int(format(i, f"0{n_qubits}b")[::-1], 2) for i in range(2**n_qubits)]
        else:
            dev_backend = "default.mixed" if noise_model != "none" else "default.qubit"
            self.dev = qml.device(dev_backend, wires=n_qubits)
            self._circuit = self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        n_features = self.n_features
        noise_model = self.noise_model
        p_depol = self.p_depol
        p_damping = self.p_damping

        @qml.qnode(self.dev, interface="numpy")
        def state_circuit(params, x):
            qml.AngleEmbedding(x, wires=range(min(n_features, n_qubits)))
            p = params.reshape(n_layers, n_qubits, 2)
            
            # Apply ansatz with noise matching models.py
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(p[layer, q, 0], wires=q)
                    qml.RZ(p[layer, q, 1], wires=q)
                    if noise_model == "depolarizing":
                        if p_depol > 0:
                            qml.DepolarizingChannel(p_depol, wires=q)
                        if p_damping > 0:
                            qml.PhaseDamping(p_damping, wires=q)
                for q in range(n_qubits):
                    ctrl = q
                    target = (q + 1) % n_qubits
                    qml.CNOT(wires=[ctrl, target])
                    if noise_model == "depolarizing":
                        if p_depol > 0:
                            qml.DepolarizingChannel(p_depol, wires=ctrl)
                            qml.DepolarizingChannel(p_depol, wires=target)
            return qml.state()

        return state_circuit

    def extract(self, X: torch.Tensor, params_flat: np.ndarray) -> np.ndarray:
        """
        Extract quantum state (statevector or density matrix) for each input sample.
        """
        if self.backend == "qiskit.aer":
            import qiskit
            params_np = params_flat.astype(float)
            states = []
            
            # Construct a batch of Qiskit circuits
            circuits = []
            for xi in X.numpy():
                qc = qiskit.QuantumCircuit(self.n_qubits)
                # AngleEmbedding: default is RX rotation
                for i in range(min(self.n_features, self.n_qubits)):
                    qc.rx(float(xi[i]), i)
                
                p = params_np.reshape(self.n_layers, self.n_qubits, 2)
                for layer in range(self.n_layers):
                    for q in range(self.n_qubits):
                        qc.ry(p[layer, q, 0], q)
                        qc.rz(p[layer, q, 1], q)
                    for q in range(self.n_qubits):
                        qc.cx(q, (q + 1) % self.n_qubits)
                
                if self.noise_model != "none":
                    qc.save_density_matrix()
                else:
                    qc.save_statevector()
                circuits.append(qc)
            
            # Run transpilation and simulation as a batch
            circuits_tr = qiskit.transpile(circuits, self.sim)
            job = self.sim.run(circuits_tr)
            results = job.result()
            
            for i in range(len(X)):
                data = results.data(i)
                if self.noise_model != "none":
                    dm = np.asarray(data["density_matrix"])
                    # Apply bit-reversal mapping for rows and columns
                    dm_pl = dm[self.idx_map][:, self.idx_map]
                    states.append(dm_pl)
                else:
                    sv = np.asarray(data["statevector"])
                    sv_pl = sv[self.idx_map]
                    states.append(sv_pl)
            return np.array(states)
        else:
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
    model = VQCClassifier(
        n_qubits=n_qubits, n_layers=n_layers, n_features=cfg.n_features,
        noise_model=cfg.noise_model, p_depol=cfg.p_depol, p_damping=cfg.p_damping,
        backend=cfg.backend
    )
    model.set_params_flat(params_pt)

    theta = params_pt.numpy().copy()  # (N,) trained parameters

    # Load standard VQC weights if available
    standard_weights_path = os.path.join(cfg.save_dir, "standard_weights.pt")
    if os.path.exists(standard_weights_path):
        standard_checkpoint = torch.load(standard_weights_path, weights_only=True)
        theta_standard = standard_checkpoint["params"].numpy().copy()
    else:
        theta_standard = None

    # Load test data using saved scaler and PCA (Issue #3)
    X_te, y_te = load_dataset_deploy(cfg)

    # Build statevector extractor once (Issue #6)
    sv_extractor = StatevectorExtractor(
        n_qubits, n_layers, cfg.n_features,
        noise_model=cfg.noise_model, p_depol=cfg.p_depol, p_damping=cfg.p_damping,
        backend=cfg.backend
    )

    # ── Branch 1: Ideal (no quantization) ──────────────────────────
    print("  [1/3] Ideal GSQ deployment (continuous params)...")
    outputs_ideal = get_outputs(model, X_te, theta)
    states_ideal  = sv_extractor.extract(X_te, theta)
    acc_ideal     = accuracy_from_outputs(outputs_ideal, y_te)
    print(f"        Accuracy: {acc_ideal:.3f}")

    # ── Branch 0: Standard VQC (m0: pure task loss + grid rounding) ──────
    if theta_standard is not None:
        print(f"  [0/3] Standard VQC grid rounding (K={cfg.K})...")
        theta_m0 = grid_round(theta_standard, cfg.K)
        outputs_m0 = get_outputs(model, X_te, theta_m0)
        states_m0  = sv_extractor.extract(X_te, theta_m0)
        acc_m0     = accuracy_from_outputs(outputs_m0, y_te)
        D_FS_m0    = fubini_study_distortion(states_ideal, states_m0)
        delta_m0   = deployment_shock(outputs_ideal, outputs_m0)
        print(f"        Accuracy: {acc_m0:.3f}  D_FS: {D_FS_m0:.4f}  Delta: {delta_m0:.4f}")
    else:
        acc_m0, D_FS_m0, delta_m0 = 0.0, 0.0, 0.0

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
        "m0":        {"accuracy": acc_m0,      "D_FS": D_FS_m0,       "delta": delta_m0},
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

    print(f"\n  {'─'*55}")
    print(f"  {'Method':<18} {'Accuracy':>10} {'D_FS':>10} {'Δ (shock)':>10}")
    print(f"  {'─'*55}")
    if theta_standard is not None:
        print(f"  {'Standard VQC (m0)':<18} {acc_m0:>10.3f} {D_FS_m0:>10.4f} {delta_m0:>10.4f}")
    print(f"  {'Ideal (GSQ)':<18} {acc_ideal:>10.3f} {'—':>10} {'—':>10}")
    print(f"  {'Euclidean (m1)':<18} {acc_euclid:>10.3f} {D_FS_euclid:>10.4f} {delta_euclid:>10.4f}")
    print(f"  {'GSQ (m2, ours)':<18} {acc_gsq:>10.3f} {D_FS_gsq:>10.4f} {delta_gsq:>10.4f}")
    print(f"  {'─'*55}")
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
    parser.add_argument("--dataset",            type=str,   default=None)
    args = parser.parse_args()

    cfg = GSQConfig.from_args(args)
    deploy(cfg)
