import numpy as np
import subprocess
import os
import scipy.sparse


def get_topK_botK_Amat(
    n_qubit: int,
    rho_vec: np.ndarray,
    K: float,
    is_dual: bool,
    is_random: bool,
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
            str(int(is_dual)),
            str(int(is_random)),
            rho_vec_filename,
            output_filename,
        ],
        stderr=subprocess.PIPE,
    ) as p:
        if verbose:
            for line in p.stderr:
                print(line.decode("utf-8", errors="ignore"), end="")
        p.wait()
    os.remove(rho_vec_filename + ".npz")
    assert p.returncode == 0, f"error in C++ code: {p.returncode=}"
    if verbose:
        print("finish to calculate with C++")

    assert os.path.exists(output_filename + ".npz")
    result = np.load(output_filename + ".npz")
    rows = result["Amat_rows"]
    data = result["Amat_data"]
    del result
    # os.remove(output_filename + ".npz")

    if rows.size == data.size == 1 and rows[0] == -1 and data[0] == 0:
        rows = np.array([])
        data = np.array([])
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
    from exputils.dot.dot_product import compute_all_dot_products

    # compile C++
    print("start to compile C++ code")
    subprocess.run(
        [
            "g++",
            "exputils/dot/fast_dot_products.cpp",
            "-o",
            "exputils/dot/fast_dot_products.exe",
            "-std=c++17",
            "-lz",
            "-Wall",
            "-Wextra",
            # "-D_GLIBCXX_DEBUG",
            "-DNDEBUG",
            "-mtune=native",
            "-march=native",
            "-O2",
            "-fopenmp",
        ]
    )
    print("finish to compile C++ code")

    # n_qubit = 3
    # K = 1.0
    # n_qubit = 4
    # K = 0.1
    # n_qubit = 5
    # K = 0.01
    n_qubit = 6
    K = 0.001
    # n_qubit = 7
    # K = 0.00001
    # n_qubit = 8
    # K = 0.00000001

    rho_vec = make_random_quantum_state("mixed", n_qubit, seed=0)

    Amat = get_topK_botK_Amat(n_qubit, rho_vec, K, is_dual=True, is_random=False)

    if n_qubit <= 5:
        dots = compute_all_dot_products(n_qubit, rho_vec)
        plt.hist([rho_vec.T @ Amat, dots], bins=100, label=["fast", "actual"])
    else:
        plt.hist(rho_vec.T @ Amat, bins=100, label=["fast"])

    plt.title("histogram of rho_vec.T @ Amat")
    plt.yscale("log")
    plt.legend()
    plt.savefig("temp.png")
    plt.show()


if __name__ == "__main__":
    main()
