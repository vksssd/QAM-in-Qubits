"""
data.py
-------
Unified data loading module for the GSQ project.

Solves two problems:
  1. Eliminates duplicated load_two_moons() across train.py and deploy.py.
  2. Persists the fitted MinMaxScaler so deployment uses exactly the same
     feature scaling as training — critical for QPU deployment correctness.
"""

import os
import numpy as np
import torch
import joblib
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_two_moons_train(
    n_samples: int = 200,
    noise: float = 0.15,
    test_size: float = 0.25,
    random_state: int = 42,
    save_dir: str = "results",
):
    """
    Load Two-Moons dataset for training.

    Fits MinMaxScaler to [0, pi] and saves it to disk so deploy can reuse it.
    Labels: {0,1} -> {-1,+1} for Pauli-Z expectation matching.

    Returns:
        (X_tr, X_te, y_tr, y_te) as float32 torch tensors.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise,
                      random_state=random_state)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    # Persist the fitted scaler
    os.makedirs(save_dir, exist_ok=True)
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    y = 2 * y - 1  # {0,1} -> {-1,+1}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.float32),
    )


def load_two_moons_deploy(
    n_samples: int = 200,
    noise: float = 0.15,
    test_size: float = 0.25,
    random_state: int = 42,
    save_dir: str = "results",
):
    """
    Load Two-Moons test set for deployment.

    Uses the SAVED scaler from training (never re-fits).
    This ensures QPU inputs are scaled identically to training.

    Returns:
        (X_te, y_te) — X_te as float32 torch tensor, y_te as float numpy array.

    Raises:
        FileNotFoundError if scaler has not been saved by a prior training run.
    """
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Fitted scaler not found at {scaler_path}. "
            f"Run train.py first to fit and save the scaler."
        )

    scaler = joblib.load(scaler_path)

    X, y = make_moons(n_samples=n_samples, noise=noise,
                      random_state=random_state)

    # Use transform(), NOT fit_transform()
    X = scaler.transform(X)

    y = 2 * y - 1  # {0,1} -> {-1,+1}

    _, X_te, _, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        torch.tensor(X_te, dtype=torch.float32),
        y_te.astype(float),
    )
