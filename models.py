"""
models.py
---------
Hardware-efficient variational quantum circuit (VQC) ansatz for GSQ experiments.

Architecture: alternating Ry/Rz rotation layers + CNOT entanglement rings.
Designed to be shallow enough for real QPU execution (4-8 qubits, 2-4 layers).
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


def build_device(n_qubits: int, backend: str = "default.qubit"):
    return qml.device(backend, wires=n_qubits)


def hardware_efficient_ansatz(params: torch.Tensor, n_qubits: int, n_layers: int):
    """
    Hardware-efficient ansatz:
      - Layer: Ry(theta) on every qubit
      - Layer: Rz(phi)  on every qubit
      - Entanglement: CNOT ring (qubit i -> qubit i+1 mod n)
    params shape: (n_layers, n_qubits, 2)  — last dim = [theta, phi]
    """
    for layer in range(n_layers):
        for q in range(n_qubits):
            qml.RY(params[layer, q, 0], wires=q)
            qml.RZ(params[layer, q, 1], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])


class VQCClassifier(nn.Module):
    """
    Binary classifier wrapping a PennyLane VQC.
    Observable: Pauli-Z on qubit 0.
    Output: expectation value in [-1, 1], thresholded at 0 for class prediction.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, n_features: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features

        # Trainable parameters: shape (n_layers, n_qubits, 2)
        n_params = n_layers * n_qubits * 2
        init = torch.zeros(n_params).uniform_(-np.pi, np.pi)
        self.params = nn.Parameter(init)

        self.dev = build_device(n_qubits)
        self.qnode = self._build_qnode()

    def _build_qnode(self):
        dev = self.dev
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        n_features = self.n_features

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(params, x):
            # Encode input via AngleEmbedding (first n_features qubits)
            qml.AngleEmbedding(x, wires=range(min(n_features, n_qubits)))
            # Variational ansatz
            p = params.reshape(n_layers, n_qubits, 2)
            hardware_efficient_ansatz(p, n_qubits, n_layers)
            return qml.expval(qml.PauliZ(0))

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_features)
        returns: (batch,) expectation values

        Uses PennyLane parameter broadcasting for batched execution
        when supported, with automatic fallback to sequential loop.
        """
        try:
            # Attempt batched execution via PennyLane broadcasting.
            # default.qubit supports this — x broadcasts over the batch dim
            # while params remain fixed.
            results = self.qnode(self.params, x)
            if results.ndim == 0:
                # Single sample, wrap in 1D tensor
                return results.unsqueeze(0)
            return results
        except Exception:
            # Fallback: sequential execution (always correct)
            return torch.stack([self.qnode(self.params, xi) for xi in x])

    def get_params_shaped(self) -> torch.Tensor:
        """Returns params as (n_layers, n_qubits, 2)."""
        return self.params.reshape(self.n_layers, self.n_qubits, 2)

    def set_params_flat(self, flat: torch.Tensor):
        """Set parameters from a flat tensor."""
        with torch.no_grad():
            self.params.copy_(flat)

    def n_total_params(self) -> int:
        return self.n_layers * self.n_qubits * 2
