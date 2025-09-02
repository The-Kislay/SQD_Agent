from __future__ import annotations
import typer
from typing import Optional

from .compare import run_sqd_benchmark
from .chemistry import rhf_build, casci_integrals_full
from .ansatz import build_ucj, build_lucj_proxy, build_he, build_hf
from .runner import run_sqd_once


app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    geom: str = typer.Option(..., help="XYZ-style string, e.g. 'Li 0 0 0; H 0 0 1.6'"),
    basis: str = typer.Option("sto-3g"),
    ansatz: str = typer.Option("ucj", help="ucj | lucj | he | hf"),
    shots: int = 300_000,
    samples_per_batch: int = 300,
    max_iterations: int = 6,
    he_layers: int = 2,
):
    """Run a single SQD calculation."""
    mol, mf = rhf_build(geom, basis)
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec

    h1, h2, e_core, _ = casci_integrals_full(mf, norb, nelec)
    if ansatz == "ucj":
        from .chemistry import ccsd_energy_and_t2
        _, t2 = ccsd_energy_and_t2(mf)
        qc = build_ucj(norb, nelec, t2)
    elif ansatz == "lucj":
        from .chemistry import ccsd_energy_and_t2
        _, t2 = ccsd_energy_and_t2(mf)
        qc = build_lucj_proxy(norb, nelec, t2, k_occ=1, k_vir=1)
    elif ansatz == "he":
        qc = build_he(norb, nelec, layers=he_layers, seed=7)
    else:
        qc = build_hf(norb, nelec)
    qc = qc.copy(); qc.measure_all()

    e_total, _ = run_sqd_once(
        h1, h2, e_core, norb, nelec, qc,
        shots=shots, samples_per_batch=samples_per_batch,
        max_iterations=max_iterations, verbose=True, label=f"SQD ({ansatz})",
    )
    typer.echo(f"\nFinal SQD energy ({ansatz}): {e_total:.8f} Ha")


@app.command()
def bench(
    geom: str = typer.Option(..., help="XYZ-style string"),
    basis: str = typer.Option("sto-3g"),
    ansatz: str = typer.Option("all", help="ucj | lucj | he | hf | all"),
    shots: int = 300_000,
    samples_per_batch: int = 300,
    max_iterations: int = 6,
    he_layers: int = 2,
    n_act_orb: Optional[int] = None,
):
    """Run the comparison table across ansätze (full & active)."""
    run_sqd_benchmark(
        atom_string=geom,
        basis=basis,
        ansatz=ansatz,
        max_iterations=max_iterations,
        shots=shots,
        samples_per_batch=samples_per_batch,
        he_layers=he_layers,
        n_act_orb=n_act_orb,
        verbose=True,
    )

def run_case(case: str):
    """Run SQD using a molecule defined in data/molecules.json"""
    cfg = get_case(case)
    run_sqd_benchmark(
        atom_string=cfg["geom"],
        basis=cfg["basis"],
        ansatz="all",
        shots=cfg["shots"],
        samples_per_batch=cfg["samples_per_batch"],
        max_iterations=cfg["max_iterations"],
        he_layers=cfg["he_layers"],
        n_act_orb=cfg["active_orbitals"],
        verbose=True,
    )

if __name__ == "__main__":
    app()
