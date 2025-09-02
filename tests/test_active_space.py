import numpy as np

from sqd.active_space import (
    choose_active_window,
    slice_t2_active_from_full,
)


def test_choose_active_window_basic():
    # closed shell: 10 spatial orbitals, 10 electrons → (5α, 5β), so ~5 occ
    norb = 10
    nelec = (5, 5)
    n_act_orb = 6

    ncore, ncas, nelecas = choose_active_window(norb, nelec, n_act_orb)

    # sanity checks
    assert 0 <= ncore <= norb
    assert 1 <= ncas <= norb
    assert ncore + ncas <= norb

    # electrons in active space are what's left after removing cores
    assert nelecas[0] == max(0, nelec[0] - ncore)
    assert nelecas[1] == max(0, nelec[1] - ncore)


def test_slice_t2_active_from_full_shape():
    # full system: nocc_full = 3, nvir_full = 3
    t2_full = np.zeros((3, 3, 3, 3))
    ncore = 1
    ncas = 4  # active_occ = 3-1 = 2; active_vir = 4-2 = 2

    t2_act = slice_t2_active_from_full(t2_full, ncore=ncore, ncas=ncas)
    assert t2_act is not None
    assert t2_act.shape == (2, 2, 2, 2)


def test_slice_t2_returns_none_when_not_feasible():
    # nocc_full = 3, nvir_full = 3
    t2_full = np.zeros((3, 3, 3, 3))
    # choose a window that yields nonpositive active_occ or active_vir
    # here active_occ = 3 - 3 = 0 → should return None
    ncore = 3
    ncas = 2

    t2_act = slice_t2_active_from_full(t2_full, ncore=ncore, ncas=ncas)
    assert t2_act is None
