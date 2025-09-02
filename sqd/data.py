import json, pathlib

_DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "molecules.json"
MOLECULES = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
DEFAULTS = MOLECULES["defaults"]

def list_molecules():
    return [m["id"] for m in MOLECULES["molecules"]]

def get_case(case_id: str):
    m = next(m for m in MOLECULES["molecules"] if m["id"].lower() == case_id.lower())
    return {**DEFAULTS, **m}
