import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix
from tqdm.auto import tqdm
from exputils.actual_Amat import get_actual_Amat, get_actual_Amats
from exputils.once.make_Amat import is_same_with_actual


def test_actual_Amat():
    print("now getting...", end=" ")
    print(get_actual_Amats())
    print("done")
    print("now checking...", end=" ")
    for n_qubit in range(1, 3 + 1):
        assert is_same_with_actual(n_qubit, get_actual_Amat(n_qubit))
    print("done")


if __name__ == "__main__":
    test_actual_Amat()
