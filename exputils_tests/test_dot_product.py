import time
import numpy as np
from tqdm.auto import tqdm
from exputils.actual_Amat import get_actual_Amat
from exputils.dot.dot_product import compute_all_dot_products
from exputils.state.random import make_random_quantum_state


def test_dot_product():
    for n_qubit in range(1, 4 + 1):
        Amat = get_actual_Amat(n_qubit)
        num_of_trial = 10 if n_qubit <= 3 else 1
        for seed in range(num_of_trial + 1):
            if seed < num_of_trial:
                rho_vec = make_random_quantum_state("mixed", n_qubit, seed)
            else:
                rho_vec = Amat[:, 0].toarray().flatten()
            slow = []
            for col in tqdm(
                Amat.T, total=Amat.shape[1], desc=f"n_qubit:{n_qubit}", leave=False
            ):
                slow.append(np.dot(rho_vec, col.toarray().flatten()))

            ans = compute_all_dot_products(n_qubit, rho_vec)
            assert np.allclose(ans, slow)

        print(f"{n_qubit=} ok!")

    rho_vec = make_random_quantum_state("mixed", 5, 0)
    t0 = time.perf_counter()
    dots = compute_all_dot_products(5, rho_vec)
    t1 = time.perf_counter()
    assert np.isclose(dots[0], get_actual_Amat(5)[:, 0].toarray().flatten() @ rho_vec)
    print(f"n_qubit:5 time:{t1-t0}[s]")

    rho_vec = make_random_quantum_state("mixed", 6, 0)
    t0 = time.perf_counter()
    compute_all_dot_products(6, rho_vec)
    t1 = time.perf_counter()
    print(f"n_qubit:6 time:{t1-t0}[s]")

    print("all ok!")


if __name__ == "__main__":
    test_dot_product()
