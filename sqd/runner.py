from __future__ import annotations
from typing import Dict, Any, Tuple, List

import numpy as np
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, SCIResult


def run_sqd_once(
    h1, h2, e_core, norb: int, nelec: Tuple[int, int], qc,
    *,
    shots: int = 300_000,
    samples_per_batch: int = 300,
    max_iterations: int = 6,
    verbose: bool = True,
    label: str = "SQD",
    print_subsamples: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Transpile + sample qc with Aer SamplerV2, then call SQD diagonalizer.
    Returns (total_energy, {"simulate": t_sim, "diag": t_diag})
    """
    import time
    from datetime import datetime

    def _now():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    backend = AerSimulator()
    tqc = transpile(qc, backend=backend, optimization_level=1)
    sampler = SamplerV2()

    if verbose:
        print(f"[{label} | simulate (shots={shots})] start   : {_now()}")
    t0 = time.time()
    job = sampler.run([tqc], shots=shots)
    meas = job.result()[0].data.meas
    t1 = time.time()
    if verbose:
        print(f"[{label} | simulate (shots={shots})] end     : {_now()}")
        print(f"[{label} | simulate (shots={shots})] duration: {t1 - t0:.3f} s\n")
    t_sim = t1 - t0

    best_e_hist: List[float] = []
    dim_hist: List[int | None] = []

    def _subspace_dim(r: SCIResult):
        try:
            amps = r.sci_state.amplitudes
            if hasattr(amps, "shape"):
                return int(np.prod(amps.shape))
        except Exception:
            pass
        for attr in ("num_configs", "n_configs", "dimension", "dim"):
            if hasattr(r.sci_state, attr):
                try:
                    return int(getattr(r.sci_state, attr))
                except Exception:
                    pass
        return None

    def callback(results: list[SCIResult]):
        r_best = min(results, key=lambda r: r.energy)
        best_e = r_best.energy + e_core
        d_best = _subspace_dim(r_best)
        best_e_hist.append(best_e)
        dim_hist.append(d_best)
        if verbose:
            d_str = f"{d_best}" if d_best is not None else "n/a"
            print(f"[{label}] Iter {len(best_e_hist):02d}: best approx = {best_e:.8f} Ha | subspace dim = {d_str}")
            if print_subsamples:
                for i, r in enumerate(results):
                    di = _subspace_dim(r)
                    ei = r.energy + e_core
                    print(f"    └─ subsample {i}: E = {ei:.8f} Ha, dim = {di if di is not None else 'n/a'}")

    if verbose:
        print(f"[{label} | SQD diagonalize] start   : {_now()}")
    t2 = time.time()
    result = diagonalize_fermionic_hamiltonian(
        h1, h2, meas,
        samples_per_batch=samples_per_batch,
        norb=norb, nelec=nelec,
        max_iterations=max_iterations,
        callback=callback,
    )
    t3 = time.time()
    if verbose:
        print(f"[{label} | SQD diagonalize] end     : {_now()}")
        print(f"[{label} | SQD diagonalize] duration: {t3 - t2:.3f} s\n")
    t_diag = t3 - t2

    # early-stop note
    iters_run = len(best_e_hist)
    if verbose and 0 < iters_run < max_iterations:
        improved = (iters_run == 1) or (abs(best_e_hist[-1] - best_e_hist[-2]) > 1e-8)
        grew = (iters_run == 1) or (dim_hist[-1] != dim_hist[-2])
        reasons = []
        if not improved:
            reasons.append("no further energy improvement")
        if not grew:
            reasons.append("subspace stopped growing")
        msg = " & ".join(reasons) if reasons else "convergence reached"
        print(f"[{label}] Early stop after {iters_run}/{max_iterations} iterations ({msg}).\n")

    e_total = result.energy + e_core
    return e_total, {"simulate": t_sim, "diag": t_diag}
