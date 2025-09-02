from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


def choose_active_window(norb: int, nelec: Tuple[int, int], n_act_orb: int):
    """
    Pick a contiguous CAS window around the Fermi level:
      returns (ncore, ncas, nelecas)
    """
    n_alpha, n_beta = nelec
    n_occ_est = (n_alpha + n_beta) // 2
    ncas = min(n_act_orb, norb)
    ncore = max(0, n_occ_est - ncas // 2)
    if ncore + ncas > norb:
        ncore = norb - ncas
    nelecas = (max(0, n_alpha - ncore), max(0, n_beta - ncore))
    return ncore, ncas, nelecas


def slice_t2_active_from_full(t2_full: np.ndarray, ncore: int, ncas: int) -> Optional[np.ndarray]:
    """
    Slice CCSD t2 (full) into the CAS window:
      active occ: [ncore : nocc_full)
      active vir: first nact_vir among virtuals
    Returns t2_active or None if not feasible.
    """
    nocc_full, _, nvir_full, _ = t2_full.shape
    nact_occ = nocc_full - ncore
    nact_vir = ncas - nact_occ
    if nact_occ <= 0 or nact_vir <= 0:
        return None
    if nact_vir > nvir_full:
        nact_vir = nvir_full
    return t2_full[ncore:nocc_full, ncore:nocc_full, :nact_vir, :nact_vir]
