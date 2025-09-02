import numpy as np
import pytest

# Skip gracefully if heavy deps aren't present
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_addon_sqd")

from sqd.ansatz import build_he  # lightweight, deterministic angles via seed
from sqd.runner import run_sqd_once


def test_runner_on_toy_hamiltonian():
    """
    Sanity-check integration:
    - Build a trivial 2-orbital Hamiltonian (h2 = 0, e_core = 0)
    - Prepare a small HE ansatz circuit
    - Run SQD with few shots / iterations
    - Verify we get a finite energy and timing dict
    """
    # --- Toy integrals in spatial-orbital form ---
    norb = 2
    nelec = (1, 1)  # spin-balanced, 2 electrons total

    # One-electron integrals: diagonal with slightly different orbital energies
    h1 = np.diag([0.5, 0.7])  # shape (norb, norb)

    # Two-electron integrals: all zeros (no interaction)
    # chemists' notation shape: (norb, norb, norb, norb)
    h2 = np.zeros((norb, norb, norb, norb))

    e_core = 0.0  # no nuclear repulsion / core shift

    # --- Small circuit: HF -> 1 layer HE (Ry + ring CX) ---
    qc = build_he(norb, nelec, layers=1, seed=123)
    qc = qc.copy()
    qc.measure_all()

    # --- Run SQD with low budgets for speed ---
    energy, timings = run_sqd_once(
        h1, h2, e_core, norb, nelec, qc,
        shots=2_000,             # low but stable enough
        samples_per_batch=20,    # small batches
        max_iterations=2,        # just a couple of SQD iterations
        verbose=False,           # keep test output clean
        label="toy",
    )

    # --- Assertions: basic sanity ---
    assert isinstance(energy, float)
    assert np.isfinite(energy)
    # energy should be in a reasonable ballpark for this toy (no huge blow-ups)
    assert abs(energy) < 10.0

    assert isinstance(timings, dict)
    assert "simulate" in timings and "diag" in timings
    assert timings["simulate"] >= 0.0
    assert timings["diag"] >= 0.0
