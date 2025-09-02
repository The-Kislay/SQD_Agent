from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
import ffsim


def build_hf(norb: int, nelec):
    """Hartree–Fock state (Jordan–Wigner) as a circuit."""
    num_qubits = 2 * norb
    qc = QuantumCircuit(num_qubits)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(num_qubits))
    return qc


def build_ucj(norb: int, nelec, t2):
    """UCJ from CCSD t2 amplitudes (spin-balanced)."""
    num_qubits = 2 * norb
    qc = QuantumCircuit(num_qubits)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(num_qubits))
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=t2)
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), range(num_qubits))
    return qc


def build_lucj_proxy(norb: int, nelec, t2, k_occ: int = 1, k_vir: int = 1):
    """
    Local-UCJ proxy: keep only 'local' doubles by |i-j|<=k_occ, |a-b|<=k_vir.
    Deterministic, no optimizer.
    """
    nocc, _, nvir, _ = t2.shape
    occ_i = np.arange(nocc)[:, None]
    occ_j = np.arange(nocc)[None, :]
    vir_a = np.arange(nvir)[:, None]
    vir_b = np.arange(nvir)[None, :]

    occ_ok = (np.abs(occ_i - occ_j) <= k_occ)
    vir_ok = (np.abs(vir_a - vir_b) <= k_vir)
    mask = occ_ok[..., None, None] & vir_ok[None, None, ...]
    t2_local = np.where(mask, t2, 0.0)

    num_qubits = 2 * norb
    qc = QuantumCircuit(num_qubits)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(num_qubits))
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=t2_local)
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), range(num_qubits))
    return qc


def build_he(norb: int, nelec, layers: int = 2, seed: int = 7):
    """
    Simple hardware-efficient ansatz:
    HF -> [Ry(θ) on all qubits + ring CX] × layers
    """
    rng = np.random.default_rng(seed)
    num_qubits = 2 * norb
    qc = QuantumCircuit(num_qubits)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(num_qubits))

    for _ in range(layers):
        thetas = rng.uniform(0.2, 1.3, size=num_qubits)
        for q in range(num_qubits):
            qc.ry(float(thetas[q]), q)
        for q in range(num_qubits):
            qc.cx(q, (q + 1) % num_qubits)
    return qc
