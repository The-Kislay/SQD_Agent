from __future__ import annotations
from typing import Tuple

import numpy as np
import pyscf
import pyscf.scf
import pyscf.mp
import pyscf.cc
import pyscf.mcscf
import pyscf.ao2mo as ao2mo
import pyscf.fci


def rhf_build(atom_string: str, basis: str):
    """Build PySCF molecule and run RHF."""
    mol = pyscf.gto.Mole()
    mol.build(atom=atom_string, basis=basis)
    mf = pyscf.scf.RHF(mol).run()
    return mol, mf


def mp2_energy(mf) -> float:
    mp2 = pyscf.mp.MP2(mf).run()
    return mf.e_tot + mp2.e_corr


def ccsd_energy_and_t2(mf):
    ccsd = pyscf.cc.CCSD(mf).run()
    return (mf.e_tot + ccsd.e_corr), ccsd.t2


def casci_integrals_full(mf, norb: int, nelec: Tuple[int, int]):
    """Return (h1, h2, e_core, e_cas) for full space."""
    mo = mf.mo_coeff
    cas = pyscf.mcscf.CASCI(mf, norb, nelec)
    h1, e_core = cas.get_h1cas(mo)
    h2 = ao2mo.restore(1, cas.get_h2cas(mo), norb)
    e_cas = cas.kernel(mo)[0]
    return h1, h2, e_core, e_cas


def casci_integrals_active(mf, ncore: int, ncas: int, nelecas: Tuple[int, int]):
    """Return (h1, h2, e_core, e_cas) for an active window with given ncore/ncas."""
    mo = mf.mo_coeff
    cas = pyscf.mcscf.CASCI(mf, ncas, nelecas)
    cas.ncore = ncore
    h1, e_core = cas.get_h1cas(mo)
    h2 = ao2mo.restore(1, cas.get_h2cas(mo), ncas)
    e_cas = cas.kernel(mo)[0]
    return h1, h2, e_core, e_cas


def fci_energy_if_feasible(h1, h2, norb, nelec, e_core, max_dets: int = 500_000):
    """Try FCI if determinant count is manageable; return (energy or None, n_dets or None)."""
    from math import comb

    n_alpha, n_beta = nelec
    try:
        dets = comb(norb, n_alpha) * comb(norb, n_beta)
    except Exception:
        return None, None
    if dets > max_dets:
        return None, dets

    e_elec, _ = pyscf.fci.direct_spin1.kernel(h1, h2, norb, nelec)
    return e_elec + e_core, dets
