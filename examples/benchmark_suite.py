"""
Batch runner: iterate several molecules (N2, LiH, CH4, BeH2, H2O, etc.)
If data/molecules.json is present, it will pull defaults from there;
otherwise uses a built-in list.

Usage:
  python examples/benchmark_suite.py
  python examples/benchmark_suite.py --cases N2_1p10A,LiH --ansatz all
"""
from __future__ import annotations
import argparse
import json
import pathlib
from typing import Dict, Any, Iterable, List

from sqd.compare import run_sqd_benchmark

DATA_JSON = pathlib.Path(__file__).resolve().parents[1] / "data" / "molecules.json"

# Fallback in case data/molecules.json is not present
FALLBACK = {
    "defaults": {
        "basis": "sto-3g",
        "shots": 300_000,
        "samples_per_batch": 300,
        "max_iterations": 6,
        "he_layers": 2,
        "active_orbitals": 6,
    },
    "molecules": [
        {"id": "N2_1p10A", "geom": "N 0 0 -0.55; N 0 0 0.55"},
        {"id": "LiH",      "geom": "Li 0 0 0; H 0 0 1.60"},
        {"id": "CH4",      "geom": "C 0 0 0; H 0 0 1.089; H 1.026719 0 -0.363; H -0.51336 -0.889165 -0.363; H -0.51336 0.889165 -0.363"},
        {"id": "BeH2",     "geom": "Be 0 0 0; H 0 0 1.316; H 0 0 -1.316"},
        {"id": "H2O",      "geom": "O 0 0 0; H 0 -0.757 0.586; H 0 0.757 0.586"},
    ],
}

def load_cases() -> Dict[str, Dict[str, Any]]:
    if DATA_JSON.exists():
        obj = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    else:
        obj = FALLBACK
    defaults = obj["defaults"]
    out: Dict[str, Dict[str, Any]] = {}
    for m in obj["molecules"]:
        cfg = {**defaults, **m}
        out[m["id"]] = cfg
    return out

def select_cases(all_cases: Dict[str, Dict[str, Any]], ids: Iterable[str]) -> List[Dict[str, Any]]:
    wanted = [i.strip() for i in ids if i.strip()]
    if not wanted:
        # take a sensible default order
        wanted = ["N2_1p10A", "LiH", "H2O"]
    selected = []
    for k in wanted:
        if k not in all_cases:
            raise SystemExit(f"Unknown case id '{k}'. Available: {', '.join(all_cases)}")
        selected.append(all_cases[k])
    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case IDs (see data/molecules.json). Default: N2_1p10A,LiH,H2O",
    )
    parser.add_argument("--ansatz", default="all", choices=["ucj", "lucj", "he", "hf", "all"])
    # allow global overrides (optional)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--samples-per-batch", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--he-layers", type=int)
    parser.add_argument("--n-act-orb", type=int)
    args = parser.parse_args()

    cases = load_cases()
    selected = select_cases(cases, args.cases.split(",") if args.cases else [])

    for cfg in selected:
        print("\n" + "#" * 84)
        print(f"# Running: {cfg.get('id', 'case')} — {cfg.get('label', cfg.get('geom')[:48])}")
        print("#" * 84 + "\n")

        run_sqd_benchmark(
            atom_string=cfg["geom"],
            basis=cfg.get("basis", "sto-3g"),
            ansatz=args.ansatz,
            shots=args.shots or cfg.get("shots", 300_000),
            samples_per_batch=args.samples_per_batch or cfg.get("samples_per_batch", 300),
            max_iterations=args.max_iterations or cfg.get("max_iterations", 6),
            he_layers=args.he_layers or cfg.get("he_layers", 2),
            n_act_orb=args.n_act_orb or cfg.get("active_orbitals", 6),
            verbose=True,
        )

if __name__ == "__main__":
    main()
