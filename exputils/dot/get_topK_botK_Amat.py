import numpy as np
import subprocess
import os
import scipy.sparse


def get_topK_botK_Amat(
    n_qubit: int,
    rho_vec: np.ndarray,
    K: float,
    verbose: bool = True,
) -> scipy.sparse.csc_matrix:
    assert 1 <= n_qubit <= 8 and 0 <= K <= 1

    rho_vec_filename = os.path.join(os.path.dirname(__file__), "temporary_rho_vec")
    output_filename = os.path.join(os.path.dirname(__file__), "temporary_result")
    np.savez_compressed(rho_vec_filename, rho_vec=rho_vec)

    if verbose:
        print("start to calculate with C++")
    with subprocess.Popen(
        [
            os.path.join(os.path.dirname(__file__), "fast_dot_products.exe"),
            str(n_qubit),
            str(K),
            rho_vec_filename,
            output_filename,
        ],
        stderr=subprocess.PIPE,
    ) as p:
        if verbose:
            for line in p.stderr:
                print(line.decode(), end="")
    os.remove(rho_vec_filename + ".npz")
    assert p.returncode == 0, f"error in C++ code: {p.returncode=}"
    if verbose:
        print("finish to calculate with C++")

    assert os.path.exists(output_filename + ".npz")
    result = np.load(output_filename + ".npz")
    rows = result["Amat_rows"]
    data = result["Amat_data"]
    del result
    os.remove(output_filename + ".npz")

    indptr = np.arange(len(rows) // (2**n_qubit) + 1) * (2**n_qubit)
    indices = rows
    Amat = scipy.sparse.csc_matrix(
        (data, indices, indptr),
        shape=(4**n_qubit, len(rows) // (2**n_qubit)),
    )
    return Amat


def main():
    import matplotlib.pyplot as plt
    from exputils.state.random import make_random_quantum_state
    from exputils.actual_Amat import get_actual_Amat

    n_qubit = 7
    K = 0.00001

    rho_vec = make_random_quantum_state("mixed", n_qubit, seed=0)

    Amat = get_topK_botK_Amat(n_qubit, rho_vec, K)

    if n_qubit <= 5:
        Amat_actual = get_actual_Amat(n_qubit)
        plt.hist(
            [rho_vec.T @ Amat, rho_vec.T @ Amat_actual],
            bins=100,
            label=["fast", "actual"],
        )
    else:
        plt.hist(rho_vec.T @ Amat, bins=100, label=["fast"])

    plt.title("histogram of rho_vec.T @ Amat")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
