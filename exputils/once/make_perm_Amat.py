import pickle

import numpy as np
from exputils.math.popcount import popcount
from exputils.perm_Amat import get_row_info
from exputils.stabilizer_group import generator_to_group, tableau_to_pauli_strs
from scipy.sparse import csc_matrix, save_npz
from tqdm import tqdm

# stabilizer_simulator 直下での実行を想定


def XA_to_tableau(n, k, XA):
    XZ_tableau = np.zeros((n, 2 * n), dtype=np.int_)
    XZ_tableau[:, :k] = XA.T
    XZ_tableau[k:n, k:n] = np.eye(n - k)
    XZ_tableau[:k, n : n + k] = np.eye(k)
    XZ_tableau[:k, n + k :] = XA[:, k:]
    return XZ_tableau


hash_root = 10007
hash_mod = 10**9 + 7


def hash_column(col):
    ret = 0
    for v in col:
        ret *= hash_root
        ret += v
        ret %= hash_mod
    return ret


class hashable_column:
    def __init__(self, col):
        self.col = col
        self.hs = hash_column(col)

    def __hash__(self):
        return self.hs

    def __eq__(self, other):
        return self.col == other.col


def make_col_info_and_perm_Amat(n, n_k_list):
    basis = get_row_info(n)
    row_size = len(basis)
    basis_index = {e: i for i, e in enumerate(basis)}

    col_info = []
    cols = []
    rows = []
    data = []
    col_size = 0
    hashed_columns = set()
    for k in range(n + 1):
        XAs = n_k_list[k]
        for XA in tqdm(XAs, desc=f"{n=}, {k=}"):
            tableau = XA_to_tableau(n, k, XA)
            generator = tableau_to_pauli_strs(tableau)
            group = generator_to_group(n, generator)
            for cidx in range(1 << n):
                col_vals = [0] * row_size
                for pidx, pauli in enumerate(group):
                    row = basis_index[
                        pauli.count("X"),
                        pauli.count("Y"),
                        pauli.count("Z"),
                    ]
                    val = (1 if popcount(cidx & pidx) % 2 == 0 else -1) * (
                        1 if pauli[0] != "-" else -1
                    )
                    col_vals[row] += val

                hs_col = hashable_column(col_vals)
                if hs_col in hashed_columns:
                    continue
                hashed_columns.add(hs_col)

                signed_generator = []
                for i, pauli in enumerate(generator):
                    if cidx & (1 << i):
                        pauli = "-" + pauli
                    signed_generator.append(pauli)
                col_info.append(signed_generator)
                for row, val in enumerate(col_vals):
                    if val:
                        cols.append(col_size)
                        rows.append(row)
                        data.append(val)
                col_size += 1
    perm_Amat = csc_matrix(
        (data, (rows, cols)), shape=(row_size, col_size), dtype=np.int16
    )
    return col_info, perm_Amat


def main():
    for n in range(1, 6 + 1):
        XA_file = f"data/permdata/X_A_{n}.npz"
        with open(XA_file, "rb") as f:
            t = np.load(f)
            n_k_list = [t[str(k)] for k in range(n + 1)]
        col_info, perm_Amat = make_col_info_and_perm_Amat(n, n_k_list)

        col_info_file = f"data/permdata/col_info_{n}.pkl"
        with open(col_info_file, "wb") as f:
            pickle.dump(col_info, f)

        perm_Amat_file = f"data/permdata/perm_Amat_{n}.npz"
        save_npz(perm_Amat_file, perm_Amat)

        print(f"{n = }, {perm_Amat.shape = }")


if __name__ == "__main__":
    main()
