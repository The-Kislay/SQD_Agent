from __future__ import annotations
from typing import Optional, Dict, Any, List
import time
from datetime import datetime

from .chemistry import (
    rhf_build,
    mp2_energy,
    ccsd_energy_and_t2,
    casci_integrals_full,
    casci_integrals_active,
    fci_energy_if_feasible,
)
from .ansatz import build_hf, build_ucj, build_lucj_proxy, build_he
from .active_space import choose_active_window, slice_t2_active_from_full
from .runner import run_sqd_once


def _now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _time(label, fn, *args, **kwargs):
    print(f"[{label}] start   : {_now()}")
    t0 = time.time()
    out = fn(*args, **kwargs)
    t1 = time.time()
    print(f"[{label}] end     : {_now()}")
    print(f"[{label}] duration: {t1 - t0:.3f} s\n")
    return out, (t1 - t0)


def _fmt_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "-+-".join("-" * w for w in widths)
    def row(cells): return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    return "\n".join([row(headers), line, *[row(r) for r in rows]])


def run_sqd_benchmark(
    atom_string: str,
    basis: str,
    ansatz: str = "ucj",              # "ucj" | "lucj" | "he" | "hf" | "all"
    max_iterations: int = 6,
    shots: int = 300_000,
    samples_per_batch: int = 300,
    he_layers: int = 2,
    n_act_orb: Optional[int] = None,
    lucj_k_occ: int = 1,
    lucj_k_vir: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    print(f"=== RUN START: {_now()} ===\n")
    print("Input:")
    print(f"  Molecule:\n{atom_string}")
    print(f"  Basis: {basis}")
    print(f"  Ansatz: {ansatz}")
    print(f"  SQD iterations: {max_iterations}, shots: {shots}, samples_per_batch: {samples_per_batch}\n")

    # SCF
    (mol, mf), t_scf = _time("RHF/SCF", rhf_build, atom_string, basis)
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    if verbose:
        print(f"Number of spatial orbitals = {norb}")
        print(f"Number of qubits (full)    = {2*norb}\n")

    # MP2, CCSD, CASCI(full)
    e_mp2, t_mp2 = _time("MP2", mp2_energy, mf)
    (e_ccsd, t2_full), t_ccsd = _time("CCSD", ccsd_energy_and_t2, mf)
    (h1_full, h2_full, e_core_full, e_cas_full), t_cas_full = _time(
        "CASCI (full-space)", casci_integrals_full, mf, norb, nelec
    )

    # FCI reference if feasible; else CASCI(full)
    e_fci, dets = fci_energy_if_feasible(h1_full, h2_full, norb, nelec, e_core_full)
    if e_fci is not None:
        ref_name, e_ref = "FCI", e_fci
        print(f"[FCI] feasible (≈{dets} dets)")
        # sanity check vs CASCI
        if abs(e_ref - e_cas_full) > 1e-4:
            print("[Warn] FCI and CASCI(full) differ by > 0.1 mHa; using CASCI(full) as reference.")
            ref_name, e_ref = "CASCI(full)", e_cas_full
        t_fci = None  # we didn't time inside helper
    else:
        print(f"[FCI] Skipped (estimated determinants ≈ {dets}).")
        ref_name, e_ref, t_fci = "CASCI(full)", e_cas_full, None

    # Active-space selection + integrals
    if n_act_orb is None:
        n_act_orb = min(norb, 6)
    ncore, ncas, nelecas = choose_active_window(norb, nelec, n_act_orb)
    if verbose:
        print(f"Active-space: ncore={ncore}, ncas={ncas}, nelecas={nelecas}\n")

    (h1_act, h2_act, e_core_act, e_cas_act), t_cas_act = _time(
        "CASCI (active-space)", casci_integrals_active, mf, ncore, ncas, nelecas
    )
    t2_active = slice_t2_active_from_full(t2_full, ncore, ncas)

    # which ansatz/zes
    ansatz_list = ["ucj", "lucj", "he", "hf"] if ansatz.lower() == "all" else [ansatz.lower()]
    bad = [a for a in ansatz_list if a not in {"ucj", "lucj", "he", "hf"}]
    if bad:
        raise ValueError(f"invalid ansatz: {bad}")

    # run SQD for each ansatz (full & active)
    rows = [
        ["RHF",              f"{mf.e_tot:.8f}",      f"{(mf.e_tot - e_ref)*1e3:+.3f}",  f"{t_scf:.3f}"],
        ["MP2",              f"{e_mp2:.8f}",         f"{(e_mp2 - e_ref)*1e3:+.3f}",     f"{t_mp2:.3f}"],
        ["CCSD",             f"{e_ccsd:.8f}",        f"{(e_ccsd - e_ref)*1e3:+.3f}",    f"{t_ccsd:.3f}"],
        ["CASCI (full)",     f"{e_cas_full:.8f}",    f"{(e_cas_full - e_ref)*1e3:+.3f}",f"{t_cas_full:.3f}"],
    ]
    if e_fci is not None:
        rows.append(["FCI (full)", f"{e_fci:.8f}", f"{(e_fci - e_ref)*1e3:+.3f}", "—"])

    results: Dict[str, Any] = {"sqd": {}}

    for a in ansatz_list:
        # full
        if a == "ucj":
            qc_full = build_ucj(norb, nelec, t2_full)
        elif a == "lucj":
            qc_full = build_lucj_proxy(norb, nelec, t2_full, k_occ=1, k_vir=1)
        elif a == "he":
            qc_full = build_he(norb, nelec, layers=he_layers, seed=7)
        else:
            qc_full = build_hf(norb, nelec)
        qc_full = qc_full.copy(); qc_full.measure_all()

        e_full, tparts_full = run_sqd_once(
            h1_full, h2_full, e_core_full, norb, nelec, qc_full,
            shots=shots, samples_per_batch=samples_per_batch,
            max_iterations=max_iterations, verbose=verbose,
            label=f"SQD (full-space, {a})",
        )
        t_full = tparts_full["simulate"] + tparts_full["diag"]

        # active (UCJ/LUCJ fallback to HE if t2_active None)
        active_label = a
        if a == "ucj":
            if t2_active is None:
                active_label = "ucj (fallback)"
                qc_act = build_he(ncas, nelecas, layers=2, seed=19)
            else:
                qc_act = build_ucj(ncas, nelecas, t2_active)
        elif a == "lucj":
            if t2_active is None:
                active_label = "lucj (fallback)"
                qc_act = build_he(ncas, nelecas, layers=2, seed=19)
            else:
                qc_act = build_lucj_proxy(ncas, nelecas, t2_active, k_occ=1, k_vir=1)
        elif a == "he":
            qc_act = build_he(ncas, nelecas, layers=he_layers, seed=19)
        else:
            qc_act = build_hf(ncas, nelecas)
        qc_act = qc_act.copy(); qc_act.measure_all()

        e_act, tparts_act = run_sqd_once(
            h1_act, h2_act, e_core_act, ncas, nelecas, qc_act,
            shots=shots, samples_per_batch=samples_per_batch,
            max_iterations=max_iterations, verbose=verbose,
            label=f"SQD (active-space, {active_label})",
        )
        t_act = tparts_act["simulate"] + tparts_act["diag"]

        rows.append([f"SQD (full) [{a}]",          f"{e_full:.8f}", f"{(e_full - e_ref)*1e3:+.3f}", f"{t_full:.3f}"])
        rows.append([f"SQD (active) [{active_label}]", f"{e_act:.8f}",  f"{(e_act - e_ref)*1e3:+.3f}",  f"{t_act:.3f}"])

        results["sqd"][a] = {
            "full": {"energy": e_full, "runtime": t_full},
            "active": {"energy": e_act, "runtime": t_act, "label": active_label},
        }

    rows.append(["CASCI (active)", f"{e_cas_act:.8f}", f"{(e_cas_act - e_ref)*1e3:+.3f}", f"{t_cas_act:.3f}"])

    print("\n=== Energy & Time Comparison ===")
    print(_fmt_table(["Method", "Energy (Ha)", f"Δ vs {ref_name} (mHa)", "Runtime (s)"], rows))
    print(f"\nReference used: {ref_name}")
    print(f"Active-space window: ncore={ncore}, ncas={ncas}, nelecas={nelecas}")
    print(f"\n=== RUN END: {_now()} ===")

    results.update({
        "reference": {"name": ref_name, "energy": e_ref},
        "energies": {
            "RHF": mf.e_tot,
            "MP2": e_mp2,
            "CCSD": e_ccsd,
            "CASCI_full": e_cas_full,
            "FCI_full": e_fci,
            "CASCI_active": e_cas_act,
        },
        "active_space": {"ncore": ncore, "ncas": ncas, "nelecas": nelecas},
        "norb_full": norb,
        "nelec_full": nelec,
        "ansatz_run": ansatz_list,
        "timings": {
            "SCF": t_scf, "MP2": t_mp2, "CCSD": t_ccsd,
            "CASCI_full": t_cas_full, "CASCI_active": t_cas_act,
            "FCI_full": t_fci,
        }
    })
    return results
