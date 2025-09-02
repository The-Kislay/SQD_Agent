"""
Quick smoke test: LiH at 1.60 Å (STO-3G).
Usage:
  python examples/run_lithium_hydride.py
  python examples/run_lithium_hydride.py --shots 200000 --ansatz ucj
"""
from __future__ import annotations
import argparse
from sqd.compare import run_sqd_benchmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom", default="Li 0 0 0; H 0 0 1.60")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--ansatz", default="ucj", choices=["ucj", "lucj", "he", "hf", "all"])
    parser.add_argument("--shots", type=int, default=300_000)
    parser.add_argument("--samples-per-batch", type=int, default=300)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--he-layers", type=int, default=2)
    parser.add_argument("--n-act-orb", type=int, default=6)
    args = parser.parse_args()

    run_sqd_benchmark(
        atom_string=args.geom,
        basis=args.basis,
        ansatz=args.ansatz,
        shots=args.shots,
        samples_per_batch=args.samples_per_batch,
        max_iterations=args.max_iterations,
        he_layers=args.he_layers,
        n_act_orb=args.n_act_orb,
        verbose=True,
    )

if __name__ == "__main__":
    main()
