"""
geometry.py
-----------
The mathematical core of GSQ. Implements:
  1. Fubini-Study distance and deployment distortion (D_FS)
  2. Circular / toroidal K-means clustering on (S^1)^N
  3. Soft geometry-preserving anchor projection
  4. Deployment shock (Delta) computation

All functions are framework-agnostic (numpy) so they can be reused
in both PennyLane training and Qiskit deployment contexts.
"""

import numpy as np
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────
# 1. Fubini–Study geometry
# ─────────────────────────────────────────────

def fubini_study_distance(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Fubini-Study distance between two normalized state vectors.

    d_FS(psi, phi) = arccos(|<psi|phi>|)

    Args:
        psi, phi: complex numpy arrays of shape (2^n,), normalized.
    Returns:
        Geodesic distance in [0, pi/2].
    """
    overlap = np.abs(np.vdot(psi, phi))
    # Clamp to [0,1] to avoid arccos domain errors from floating point
    overlap = np.clip(overlap, 0.0, 1.0)
    return float(np.arccos(overlap))


def fubini_study_distortion(
    states_train: np.ndarray,
    states_deploy: np.ndarray
) -> float:
    """
    Expected Fubini-Study deployment distortion D_FS.

    D_FS = E_x[ d_FS(psi(theta, x), psi(theta_hat, x))^2 ]

    Supports both pure statevectors (ndim=2) and noisy mixed density matrices (ndim=3).
    """
    assert states_train.shape == states_deploy.shape

    if states_train.ndim == 2:
        # Pure state vectors: (n_samples, 2^n)
        overlaps = np.abs(np.einsum('ij,ij->i', states_train.conj(), states_deploy))
        overlaps = np.clip(overlaps, 0.0, 1.0)
        return float(np.mean(np.arccos(overlaps) ** 2))
    elif states_train.ndim == 3:
        # Mixed state density matrices: (n_samples, 2^n, 2^n)
        from scipy.linalg import sqrtm
        dists = []
        for i in range(len(states_train)):
            rho = states_train[i]
            sigma = states_deploy[i]
            try:
                # Bures fidelity: F = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
                sqrt_rho = sqrtm(rho)
                temp = sqrt_rho @ sigma @ sqrt_rho
                sqrt_temp = sqrtm(temp)
                fid = float(np.real(np.trace(sqrt_temp)) ** 2)
                fid = np.clip(fid, 0.0, 1.0)
                dists.append(np.arccos(np.sqrt(fid)) ** 2)
            except Exception:
                # Numerical fallback: Tr(rho * sigma)
                overlap = float(np.real(np.trace(rho @ sigma)))
                overlap = np.clip(overlap, 0.0, 1.0)
                dists.append(np.arccos(np.sqrt(overlap)) ** 2)
        return float(np.mean(dists))
    else:
        raise ValueError(f"Invalid state array dimension: {states_train.ndim}")


def deployment_shock(
    outputs_train: np.ndarray,
    outputs_deploy: np.ndarray
) -> float:
    """
    Operational deployment shock Delta.

    Delta = E_x[ (f(theta, x) - f(theta_hat, x))^2 ]

    Args:
        outputs_train:  (n_samples,) float array — trained model outputs
        outputs_deploy: (n_samples,) float array — quantized model outputs
    Returns:
        Scalar Delta value.
    """
    return float(np.mean((outputs_train - outputs_deploy) ** 2))


# ─────────────────────────────────────────────
# 2. Toroidal / Circular geometry utilities
# ─────────────────────────────────────────────

def circular_distance(a: float, b: float) -> float:
    """
    Geodesic distance on S^1.
    d_S1(a, b) = min_m |a - b + 2*pi*m|
    """
    diff = a - b
    return float(abs((diff + np.pi) % (2 * np.pi) - np.pi))


def toroidal_distance(theta: np.ndarray, center: np.ndarray) -> float:
    """
    Toroidal distance on (S^1)^N.
    d_torus(theta, c)^2 = sum_i d_S1(theta_i, c_i)^2
    """
    diff = (theta - center + np.pi) % (2 * np.pi) - np.pi
    return float(np.sqrt(np.sum(diff ** 2)))


def circular_mean(angles: np.ndarray) -> np.ndarray:
    """
    Compute the circular mean of an array of angles via unit-circle embedding.
    angles: (n_samples, N) array of angular parameters
    Returns: (N,) array of circular means, one per parameter dimension.
    """
    sin_mean = np.mean(np.sin(angles), axis=0)
    cos_mean = np.mean(np.cos(angles), axis=0)
    return np.arctan2(sin_mean, cos_mean)


# ─────────────────────────────────────────────
# 3. Circular K-Means on (S^1)^N
# ─────────────────────────────────────────────

class CircularKMeans:
    """
    K-means clustering on the N-torus (S^1)^N.

    Unlike Euclidean K-means, this:
      - Uses circular/toroidal geodesic distance for assignment
      - Uses circular mean for centroid updates
      - Avoids angular wraparound artifacts

    This directly implements Definition 4.2 from the GSQ theory section.
    """

    def __init__(self, K: int, max_iter: int = 200, n_init: int = 10,
                 tol: float = 1e-6, random_state: Optional[int] = None):
        self.K = K
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """K-means++ style init adapted for circular distance."""
        n, N = X.shape
        idx = self.rng.integers(0, n)
        centers = [X[idx].copy()]

        for _ in range(1, self.K):
            dists = np.array([
                min(toroidal_distance(x, c) for c in centers) for x in X
            ])
            probs = dists ** 2
            total = probs.sum()
            if total == 0 or np.isnan(total):
                # All points are already selected or equidistant — uniform fallback
                idx = self.rng.integers(0, n)
            else:
                probs /= total
                idx = self.rng.choice(n, p=probs)
            centers.append(X[idx].copy())

        return np.array(centers)

    def _assign(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # Vectorized pairwise toroidal distance: (n_samples, K)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
        dists = np.sqrt(np.sum(wrapped ** 2, axis=2))  # (n_samples, K)
        return np.argmin(dists, axis=1).astype(int)

    def _update_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        new_centers = np.zeros_like(self.centers_)
        for k in range(self.K):
            mask = labels == k
            if mask.sum() == 0:
                # Empty cluster: reinitialize randomly
                new_centers[k] = X[self.rng.integers(0, len(X))].copy()
            else:
                new_centers[k] = circular_mean(X[mask])
        return new_centers

    def _inertia(self, X: np.ndarray, labels: np.ndarray,
                 centers: np.ndarray) -> float:
        assigned_centers = centers[labels]  # (n_samples, N)
        diff = (X - assigned_centers + np.pi) % (2 * np.pi) - np.pi
        return float(np.sum(diff ** 2))

    def fit(self, X: np.ndarray) -> "CircularKMeans":
        """
        Fit circular K-means on angular parameter array.
        X: (n_samples, N) array of angular parameters in [-pi, pi].
        """
        # Clamp K to available samples to avoid degenerate clustering
        if len(X) < self.K:
            self.K = len(X)
        best_centers = None
        best_labels = None
        best_inertia = float("inf")

        for _ in range(self.n_init):
            centers = self._init_centers(X)

            for iteration in range(self.max_iter):
                labels = self._assign(X, centers)
                self.centers_ = centers
                new_centers = self._update_centers(X, labels)

                diff_c = (new_centers - centers + np.pi) % (2 * np.pi) - np.pi
                shift = float(np.max(np.sqrt(np.sum(diff_c ** 2, axis=1))))
                centers = new_centers
                if shift < self.tol:
                    break

            labels = self._assign(X, centers)
            inertia = self._inertia(X, labels, centers)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest anchor."""
        assert self.centers_ is not None, "Call fit() first."
        return self._assign(X, self.centers_)

    def nearest_anchor(self, theta: np.ndarray) -> np.ndarray:
        """Return the nearest anchor center for a single parameter vector."""
        assert self.centers_ is not None, "Call fit() first."
        diff = (theta - self.centers_ + np.pi) % (2 * np.pi) - np.pi
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        return self.centers_[int(np.argmin(dists))].copy()


# ─────────────────────────────────────────────
# 4. Geometric projection
# ─────────────────────────────────────────────

def soft_geometric_projection(
    theta: np.ndarray,
    anchor: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Soft geometry-preserving anchor relaxation (Phase III of GSQ).

    theta_new = alpha * theta + (1 - alpha) * anchor*

    Interpolation is done on the circle to respect toroidal geometry:
    we slerp each dimension independently via angle wrapping.

    Args:
        theta:  (N,) current parameter vector
        anchor: (N,) nearest anchor center
        alpha:  blend coefficient in [0,1]. 1.0 = no projection, 0.0 = full snap.
    Returns:
        (N,) blended parameter vector.
    """
    diff = (anchor - theta + np.pi) % (2 * np.pi) - np.pi
    result = theta + (1.0 - alpha) * diff
    # Wrap back to [-pi, pi]
    result = (result + np.pi) % (2 * np.pi) - np.pi
    return result


def hard_quantize(theta: np.ndarray, kmeans: CircularKMeans) -> np.ndarray:
    """
    Final deployment quantization: snap to nearest anchor.
    theta_hat = argmin_{c_k in C} d_torus(theta, c_k)
    """
    return kmeans.nearest_anchor(theta)


def quantize_params(
    theta: np.ndarray,
    kmeans: CircularKMeans,
    alpha: float = 0.5,
    n_relaxation_steps: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Phase III + IV pipeline:
      1. Iterative soft projection toward nearest anchor
      2. Hard quantization to nearest anchor

    Args:
        theta:               (N,) trained parameter vector
        kmeans:              fitted CircularKMeans instance
        alpha:               soft projection blend (0.5 default)
        n_relaxation_steps:  how many soft projection iterations before hard snap

    Returns:
        theta_relaxed: (N,) after soft projection
        theta_hat:     (N,) after hard quantization
    """
    theta_relaxed = theta.copy()
    for _ in range(n_relaxation_steps):
        anchor = kmeans.nearest_anchor(theta_relaxed)
        theta_relaxed = soft_geometric_projection(theta_relaxed, anchor, alpha)

    theta_hat = hard_quantize(theta_relaxed, kmeans)
    return theta_relaxed, theta_hat


def euclidean_quantize(
    theta: np.ndarray,
    training_trajectory: np.ndarray,
    K: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Baseline: standard Euclidean K-means quantization (for ablation comparison).

    Fits sklearn KMeans on a real training trajectory (parameter snapshots
    recorded during training), then snaps theta to the nearest centroid.
    This is the fair Euclidean counterpart to GSQ's circular clustering.

    Args:
        theta:                (N,) trained parameter vector to quantize.
        training_trajectory:  (T, N) array of parameter snapshots from training.
        K:                    Number of cluster centroids.
        random_state:         Random seed for KMeans.
    Returns:
        (N,) quantized parameter vector (nearest Euclidean centroid).
    """
    from sklearn.cluster import KMeans

    assert training_trajectory.ndim == 2, (
        f"training_trajectory must be 2D (T, N), got shape {training_trajectory.shape}"
    )
    # Clamp K to available samples
    effective_K = min(K, len(training_trajectory))
    km = KMeans(n_clusters=effective_K, random_state=random_state, n_init=10)
    km.fit(training_trajectory)
    dists = np.linalg.norm(km.cluster_centers_ - theta, axis=1)
    return km.cluster_centers_[np.argmin(dists)].copy()
