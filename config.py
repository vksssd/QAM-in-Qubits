"""
config.py
---------
Centralized hyperparameter configuration for the GSQ project.

All scripts import defaults from here. CLI arguments override when provided.
This eliminates scattered hyperparameters across train.py and deploy.py,
preventing version-control headaches during ablation sweeps.
"""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class GSQConfig:
    """All GSQ hyperparameters in one place."""

    # ── Circuit architecture ──────────────────────────────
    n_qubits: int = 4
    n_layers: int = 2
    n_features: int = 2

    # ── Training (Phases I & II) ──────────────────────────
    epochs: int = 100
    lr: float = 0.05
    tau: float = 0.6           # Phase transition point (fraction of total epochs)
    lambda_max: float = 0.3    # Max geometric regularizer weight
    k_period: int = 2          # Periodicity of geometric regularizer
    k_rate: float = 5.0        # Exponential ramp rate for lambda schedule

    # ── Data ──────────────────────────────────────────────
    n_samples: int = 200
    test_size: float = 0.25
    noise: float = 0.15

    # ── Deployment (Phases III–V) ─────────────────────────
    K: int = 8                 # Number of quantization anchors
    alpha: float = 0.5         # Soft projection blend (1=no projection, 0=full snap)
    n_relaxation_steps: int = 3

    # ── Training trajectory (for baseline fitting) ────────
    trajectory_save_every: int = 5  # Save param snapshot every N epochs

    # ── Accuracy logging ──────────────────────────────────
    test_acc_every: int = 5    # Evaluate test accuracy every N epochs

    # ── I/O ───────────────────────────────────────────────
    save_dir: str = "results"
    fig_dir: str = "figures"
    seed: int = 42

    def save(self, path: str = None):
        """Serialize config to JSON."""
        if path is None:
            path = os.path.join(self.save_dir, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GSQConfig":
        """Load config from JSON."""
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    @classmethod
    def from_args(cls, args) -> "GSQConfig":
        """Build config from argparse namespace, using defaults for missing fields."""
        cfg = cls()
        for key in asdict(cfg):
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(cfg, key, getattr(args, key))
        return cfg


# Module-level default instance for convenience
DEFAULT = GSQConfig()
