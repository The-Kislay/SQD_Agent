import numpy as np
from qiskit import QuantumCircuit

from sqd.ansatz import (
    build_hf,
    build_ucj,
    build_lucj_proxy,
    build_he,
)


def _dummy_t2(nocc: int, nvir: int):
    # tiny, deterministic CCSD-like tensor with the correct shape
    t2 = np.zeros((nocc, nocc, nvir, nvir))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    t2[i, j, a, b] = (i == j) * 0.02 + (a == b) * 0.01
    return t2


def test_build_hf_qubits():
    norb = 4
    nelec = (2, 2)
    qc = build_hf(norb, nelec)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 2 * norb
    # HF should at least have some ops (state prep)
    assert qc.size() > 0


def test_build_ucj_qubits_and_size():
    norb = 4
    nelec = (2, 2)
    t2 = _dummy_t2(nocc=2, nvir=2)  # shape (2,2,2,2) matches norb=4, closed shell
    qc = build_ucj(norb, nelec, t2)
    assert qc.num_qubits == 2 * norb
    assert qc.size() > 0  # HF + UCJ ops


def test_build_lucj_proxy_qubits_and_size():
    norb = 4
    nelec = (2, 2)
    t2 = _dummy_t2(nocc=2, nvir=2)
    qc = build_lucj_proxy(norb, nelec, t2, k_occ=1, k_vir=1)
    assert qc.num_qubits == 2 * norb
    assert qc.size() > 0  # HF + masked UCJ ops


def test_build_he_qubits_and_size():
    norb = 4
    nelec = (2, 2)
    qc = build_he(norb, nelec, layers=2, seed=7)
    assert qc.num_qubits == 2 * norb
    assert qc.size() > 0  # HF + HE layers
