"""
train.py
--------
GSQ training pipeline — Phases I and II.

Phase I  (t < tau*T): Free exploration — task loss only.
Phase II (t >= tau*T): Geometric hardening — task loss + periodic manifold regularizer.

Saves trained weights, training trajectory, and fitted scaler to disk
for deploy.py to load.

Usage:
    python train.py --n_qubits 4 --n_layers 2 --epochs 100 --tau 0.6 --K 8
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import GSQConfig
from data import load_two_moons_train
from models import VQCClassifier


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────

def task_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE between Pauli-Z expectation values and {-1,+1} labels."""
    return nn.functional.mse_loss(outputs, labels)


def geometric_regularizer(params_flat: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Periodic manifold regularizer (Phase II).
    L_geom = (1/N) * sum_i sin^2(k * theta_i)

    Induces attractor wells on (S^1)^N at multiples of pi/k.
    Encourages parameters to settle near quantization-friendly positions.
    """
    return torch.mean(torch.sin(k * params_flat) ** 2)


def lambda_schedule(t: int, T: int, tau: float,
                    lambda_max: float, k_rate: float = 5.0) -> float:
    """
    Delayed activation schedule for geometric regularizer.
    lambda(t) = 0                              if t < tau*T
              = lambda_max*(1-exp(-k*(t-t0)/T)) if t >= tau*T
    """
    t0 = int(tau * T)
    if t < t0:
        return 0.0
    return lambda_max * (1.0 - np.exp(-k_rate * (t - t0) / T))


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def accuracy_from_outputs(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy from pre-computed outputs (avoids re-running the circuit)."""
    with torch.no_grad():
        predicted = torch.sign(outputs.detach())
        predicted[predicted == 0] = 1
    return float((predicted == labels).float().mean())


def accuracy(model: VQCClassifier, X: torch.Tensor,
             y: torch.Tensor) -> float:
    """Run a full forward pass and compute accuracy. Used for test set evaluation."""
    model.eval()
    with torch.no_grad():
        preds = model(X)
        predicted = torch.sign(preds)
        predicted[predicted == 0] = 1
    acc = float((predicted == y).float().mean())
    model.train()
    return acc


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(cfg: GSQConfig = None) -> dict:
    if cfg is None:
        cfg = GSQConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  GSQ Training  |  {cfg.n_qubits}q  {cfg.n_layers}L  {cfg.epochs} epochs")
    print(f"  tau={cfg.tau}  lambda_max={cfg.lambda_max}  k={cfg.k_period}")
    print(f"{'='*55}\n")

    # Data (saves scaler to disk)
    X_tr, X_te, y_tr, y_te = load_two_moons_train(
        n_samples=cfg.n_samples,
        noise=cfg.noise,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        save_dir=cfg.save_dir,
    )

    # Model
    model = VQCClassifier(n_qubits=cfg.n_qubits, n_layers=cfg.n_layers,
                          n_features=cfg.n_features)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    history = {
        "loss_task": [], "loss_geom": [], "loss_total": [],
        "train_acc": [], "test_acc": [], "lambda_vals": [],
    }

    # Training trajectory: save parameter snapshots for baseline fitting.
    # Ensure minimum trajectory density — at least 10 snapshots for clustering.
    trajectory_snapshots = [model.params.detach().cpu().clone()]  # initial random params
    traj_interval = max(1, min(cfg.trajectory_save_every,
                               cfg.epochs // 10))  # adaptive for short runs
    last_te_acc = 0.0  # Will be updated on first epoch (epoch 0 % N == 0)

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_tr)
        l_task = task_loss(outputs, y_tr)

        lam = lambda_schedule(epoch, cfg.epochs, cfg.tau, cfg.lambda_max,
                              k_rate=cfg.k_rate)
        l_geom = geometric_regularizer(model.params, k=cfg.k_period)
        l_total = l_task + lam * l_geom

        l_total.backward()
        optimizer.step()

        # ── Save trajectory snapshot ──
        if epoch % traj_interval == 0 or epoch == cfg.epochs - 1:
            trajectory_snapshots.append(model.params.detach().cpu().clone())

        # ── Logging (Issue #5: cache train outputs, Issue #10: model.eval) ──
        # Train accuracy: reuse outputs from forward pass (no extra circuit cost)
        tr_acc = accuracy_from_outputs(outputs, y_tr)

        # Test accuracy: only compute every N epochs to save circuit evaluations
        if epoch % cfg.test_acc_every == 0 or epoch == cfg.epochs - 1:
            te_acc = accuracy(model, X_te, y_te)
            last_te_acc = te_acc
        else:
            te_acc = last_te_acc

        history["loss_task"].append(float(l_task.detach()))
        history["loss_geom"].append(float(l_geom.detach()))
        history["loss_total"].append(float(l_total.detach()))
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["lambda_vals"].append(lam)

        phase = "II (hardening)" if lam > 0 else "I  (explore) "
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4d}/{cfg.epochs}  "
                  f"Phase {phase}  "
                  f"L={float(l_total.detach()):.4f}  "
                  f"Reg={float(l_geom.detach()):.4f}  "
                  f"λ={lam:.4f}  "
                  f"TestAcc={te_acc:.3f}")

    # ── Save weights, trajectory, config, and history ──
    trajectory_tensor = torch.stack(trajectory_snapshots)

    weights_path = os.path.join(cfg.save_dir, "trained_weights.pt")
    torch.save({
        "params": model.params.detach().cpu(),
        "n_qubits": cfg.n_qubits,
        "n_layers": cfg.n_layers,
        "n_features": cfg.n_features,
        "training_trajectory": trajectory_tensor,
        "hyperparams": {
            "tau": cfg.tau, "lambda_max": cfg.lambda_max,
            "k_period": cfg.k_period, "epochs": cfg.epochs,
            "seed": cfg.seed,
        },
    }, weights_path)

    history_path = os.path.join(cfg.save_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    cfg.save()

    print(f"\n  Saved weights    -> {weights_path}")
    print(f"  Saved history    -> {history_path}")
    print(f"  Trajectory size  -> {trajectory_tensor.shape[0]} snapshots")
    print(f"\n  Final test accuracy: {history['test_acc'][-1]:.3f}\n")

    return history


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSQ Training — Phases I & II")
    parser.add_argument("--n_qubits",    type=int,   default=None)
    parser.add_argument("--n_layers",    type=int,   default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--tau",         type=float, default=None)
    parser.add_argument("--lambda_max",  type=float, default=None)
    parser.add_argument("--k_period",    type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--n_samples",   type=int,   default=None)
    parser.add_argument("--save_dir",    type=str,   default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    args = parser.parse_args()

    cfg = GSQConfig.from_args(args)
    train(cfg)
