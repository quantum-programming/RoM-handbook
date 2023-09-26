import numpy as np

from exputils.actual_Amat import get_actual_Amat
from exputils.dot.tensor_type_dot import (
    _fast_argsort,
    compute_topK_tensor_type_dot_products,
)
from exputils.stabilizer_group import total_stabilizer_group_size
from exputils.state.random import make_random_quantum_state


def test_fast_argsort():
    for _ in range(10):
        sz = 100
        vals = np.random.rand(sz)
        K = np.random.randint(1, 3)
        topK_idxs_uv, botK_idxs_uv = _fast_argsort(vals, K)
        answer_top = np.argpartition(-vals, K)[:K]
        answer_bot = np.argpartition(vals, K)[:K]
        assert np.allclose(np.sort(answer_top), np.sort(topK_idxs_uv))
        assert np.allclose(np.sort(answer_bot), np.sort(botK_idxs_uv))
    for _ in range(10):
        sz = 20000
        vals = np.random.rand(sz)
        K = np.random.randint(1, sz)
        topK_idxs_uv, botK_idxs_uv = _fast_argsort(vals, K)
        answer_top = np.argpartition(-vals, K)[:K]
        answer_bot = np.argpartition(vals, K)[:K]
        assert set(topK_idxs_uv.tolist()).issubset(answer_top.tolist())
        assert set(botK_idxs_uv.tolist()).issubset(answer_bot.tolist())


def test_compute_topK_tensor_type_dot_products():
    for n in range(2, 4 + 1):
        actual_Amat = get_actual_Amat(n).T.toarray()
        for seed in range(5):
            rho_vec = make_random_quantum_state("mixed", n, seed=seed)
            K = total_stabilizer_group_size(n // 2)
            Amat, dots = compute_topK_tensor_type_dot_products(n, rho_vec, K)
            assert np.allclose(dots, rho_vec.T @ Amat)
            assert all(col in actual_Amat for col in Amat.T)
        print(f"n={n} ok!")
    print("all ok!")


if __name__ == "__main__":
    test_fast_argsort()
    test_compute_topK_tensor_type_dot_products()
