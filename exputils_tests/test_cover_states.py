import numpy as np
from tqdm.auto import tqdm
from numba import njit
from exputils.cover_states import cover_states
from exputils.actual_Amat import get_actual_Amat, generators_to_Amat
from exputils.stabilizer_group import generator_to_group_in_phase_pidx


def is_in_actual_Amat(n_qubit: int, col: np.ndarray) -> bool:
    """check if the given column is included in the actual Amat"""
    actual_Amat = get_actual_Amat(n_qubit).T
    for i in range(actual_Amat.shape[0]):
        if np.allclose(col, actual_Amat[i].toarray()):
            return True
    return False


@njit(cache=True)
def reflect_pidx_to_seen_cnt(seen_cnt, g_pidx):
    for pidx in g_pidx:
        seen_cnt[pidx] += 1


def test_cover_states():
    for n in range(1, 9):
        cover = cover_states(n)
        assert len(cover) == 2**n + 1
        seen_cnt = np.zeros(4**n, dtype=np.int32)
        for state in tqdm(cover, desc=f"n={n}", leave=False):
            _g_phase, g_pidx = generator_to_group_in_phase_pidx(n, state)
            reflect_pidx_to_seen_cnt(seen_cnt, g_pidx)
        assert seen_cnt[0] == (2**n + 1) and np.allclose(seen_cnt[1:], 1)
        if n <= 3:
            cover_generators = cover_states(n)
            cover_mat = generators_to_Amat(n, cover_generators)
            assert all(is_in_actual_Amat(n, col) for col in cover_mat.T.toarray())
        print(n, "ok")

    print("all ok")


if __name__ == "__main__":
    test_cover_states()
