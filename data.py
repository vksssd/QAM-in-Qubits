"""
data.py
-------
Unified data loading module for the GSQ project.

Solves three problems:
  1. Eliminates duplicated loading logic.
  2. Persists the fitted scaler (and PCA model for MNIST) so deployment
     uses exactly the same transformation pipeline as training.
  3. Supports both synthetic Two Moons and real-world MNIST 0 vs 1 datasets.
"""

import os
import numpy as np
import torch
import joblib
import torchvision.datasets as datasets
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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
    Labels: {0,1} -> {-1,+1}
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

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
    """
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Fitted scaler not found at {scaler_path}. Run train.py first."
        )

    scaler = joblib.load(scaler_path)

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X = scaler.transform(X)

    y = 2 * y - 1  # {0,1} -> {-1,+1}

    _, X_te, _, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        torch.tensor(X_te, dtype=torch.float32),
        y_te.astype(float),
    )


def load_mnist_train(
    n_samples: int = 200,
    n_features: int = 2,
    test_size: float = 0.25,
    random_state: int = 42,
    save_dir: str = "results",
):
    """
    Load MNIST dataset, filter for 0 and 1, downsample to n_samples,
    project to n_features dimensions using PCA, scale to [0, pi],
    and save both PCA and MinMaxScaler models to disk.
    """
    os.makedirs("./data_cache", exist_ok=True)
    mnist = datasets.MNIST(root="./data_cache", train=True, download=True)

    # Filter classes 0 and 1
    targets = mnist.targets
    mask = (targets == 0) | (targets == 1)
    X_raw = mnist.data[mask].numpy().reshape(-1, 28 * 28) / 255.0
    y_raw = targets[mask].numpy()

    # Downsample using stratified split if requested
    if 0 < n_samples < len(X_raw):
        X_sub, _, y_sub, _ = train_test_split(
            X_raw, y_raw, train_size=n_samples, random_state=random_state, stratify=y_raw
        )
        X_raw, y_raw = X_sub, y_sub

    # PCA feature reduction
    pca = PCA(n_components=n_features)
    X_pca = pca.fit_transform(X_raw)

    # Scale to toroidal space [0, pi]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_pca)

    # Save PCA and Scaler to disk
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(pca, os.path.join(save_dir, "pca.joblib"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))

    y = 2 * y_raw - 1  # {0, 1} -> {-1, 1}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.float32),
    )


def load_mnist_deploy(
    n_samples: int = 200,
    n_features: int = 2,
    test_size: float = 0.25,
    random_state: int = 42,
    save_dir: str = "results",
):
    """
    Load MNIST dataset, filter for 0 and 1, downsample, and project/scale
    using pre-saved PCA and MinMaxScaler models from training.
    """
    pca_path = os.path.join(save_dir, "pca.joblib")
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    if not os.path.exists(pca_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Fitted PCA/scaler not found in {save_dir}. Run train.py first."
        )

    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)

    os.makedirs("./data_cache", exist_ok=True)
    mnist = datasets.MNIST(root="./data_cache", train=True, download=True)

    # Filter classes 0 and 1
    targets = mnist.targets
    mask = (targets == 0) | (targets == 1)
    X_raw = mnist.data[mask].numpy().reshape(-1, 28 * 28) / 255.0
    y_raw = targets[mask].numpy()

    # Downsample
    if 0 < n_samples < len(X_raw):
        X_sub, _, y_sub, _ = train_test_split(
            X_raw, y_raw, train_size=n_samples, random_state=random_state, stratify=y_raw
        )
        X_raw, y_raw = X_sub, y_sub

    # Project and scale using saved transformations (NO fit!)
    X_pca = pca.transform(X_raw)
    X_scaled = scaler.transform(X_pca)

    y = 2 * y_raw - 1  # {0, 1} -> {-1, 1}

    _, X_te, _, y_te = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        torch.tensor(X_te, dtype=torch.float32),
        y_te.astype(float),
    )


def load_dataset_train(cfg, save_dir: str = None):
    """Unified entry point for loading training data."""
    sd = save_dir if save_dir is not None else cfg.save_dir
    if cfg.dataset == "two_moons":
        return load_two_moons_train(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            save_dir=sd,
        )
    elif cfg.dataset == "mnist":
        return load_mnist_train(
            n_samples=cfg.n_samples,
            n_features=cfg.n_features,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            save_dir=sd,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


def load_dataset_deploy(cfg, save_dir: str = None):
    """Unified entry point for loading deployment data."""
    sd = save_dir if save_dir is not None else cfg.save_dir
    if cfg.dataset == "two_moons":
        return load_two_moons_deploy(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            save_dir=sd,
        )
    elif cfg.dataset == "mnist":
        return load_mnist_deploy(
            n_samples=cfg.n_samples,
            n_features=cfg.n_features,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            save_dir=sd,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
