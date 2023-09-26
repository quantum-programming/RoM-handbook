import os
import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.io import loadmat
from tqdm import tqdm
from exputils.dot.load_data import load_dot_data
from exputils.math.fwht import sylvesters
from exputils.stabilizer_group import total_stabilizer_group_size


def make_Amat_from_column_index(
    n_qubit: int, idxs: np.ndarray, data_per_col: np.ndarray, rows_per_col: np.ndarray
) -> csc_matrix:
    q, r = np.divmod(idxs, 2**n_qubit)
    H = sylvesters(n_qubit)
    indptr = np.arange(len(idxs) + 1) * (2**n_qubit)
    indices = rows_per_col[q].flatten()
    data = data_per_col[q].flatten() * H[r].flatten().astype(np.int8)
    return csc_matrix((data, indices, indptr), shape=(4**n_qubit, len(idxs)))


def make_Amat(
    n_qubit: int, data_per_col: np.ndarray, rows_per_col: np.ndarray
) -> csc_matrix:
    """from the representation of the Amat (data_per_col, rows_per_col),
    make the sparse matrix of the Amat

    Args:
        n_qubit (int): the number of qubits
        data_per_col (np.ndarray): the phase of the Amat
        rows_per_col (np.ndarray): the index of the Amat

    Returns:
        csc_matrix: the sparse matrix of the Amat
    """
    return make_Amat_from_column_index(
        n_qubit,
        np.arange(total_stabilizer_group_size(n_qubit)),
        data_per_col,
        rows_per_col,
    )


def is_same_with_actual(n_qubit: int, calced_Amat: csc_matrix) -> bool:
    # Since the dense matrix is too large,
    # we compare the string representation of the matrix.
    if n_qubit == 1:
        mat = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, -1, 0, 0, 0, 0],
                [0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 1, -1],
            ]
        )
        actual_Amat = csc_matrix(mat)
    else:
        mat = loadmat(
            os.path.join(
                os.path.dirname(__file__), f"../../data/Amat/Amat{n_qubit}.mat"
            )
        )
        actual_Amat = csc_matrix(mat["A"])

    def f(col):
        ret = ""
        assert len(col.data) == (2**n_qubit)
        for data, index in sorted(zip(col.data, col.indices)):
            ret += str(data) + "," + str(index) + ","
        return ret

    A = [
        f(actual_Amat.getcol(i))
        for i in tqdm(range(actual_Amat.shape[1]), desc="actual", leave=False)
    ]
    A.sort()
    C = [
        f(calced_Amat.getcol(i))
        for i in tqdm(range(calced_Amat.shape[1]), desc="calced", leave=False)
    ]
    C.sort()
    print(f"{A[:5]=}")
    print(f"{C[:5]=}")
    return A == C


def main():
    import matplotlib.pyplot as plt

    for n_qubit in range(1, 5 + 1):
        print(f"n_qubit={n_qubit}")
        data_per_col, rows_per_col = load_dot_data(n_qubit)
        Amat = make_Amat(n_qubit, data_per_col, rows_per_col)
        if n_qubit <= 2:
            plt.imshow(Amat.toarray())
            plt.show()
        scipy.sparse.save_npz(
            os.path.join(
                os.path.dirname(__file__), f"../../data/Amat/Amat{n_qubit}.npz"
            ),
            Amat,
        )
        if n_qubit <= 4:
            assert is_same_with_actual(n_qubit, Amat)


if __name__ == "__main__":
    main()
