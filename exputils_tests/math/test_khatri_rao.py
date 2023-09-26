from exputils.math.khatri_rao import khatri_rao_from_Amat
from exputils.actual_Amat import get_actual_Amat
import numpy as np
import scipy.sparse
import time


def test_khatri_rao_for_Amat():
    for _ in range(10):
        Amat = get_actual_Amat(np.random.randint(1, 4 + 1))
        A = Amat[:, np.random.randint(Amat.shape[1], size=500)]
        Amat = get_actual_Amat(np.random.randint(1, 4 + 1))
        B = Amat[:, np.random.randint(Amat.shape[1], size=500)]

        t0 = time.perf_counter()
        C = []
        for i in range(A.shape[1]):
            C.append(
                scipy.sparse.kron(
                    A[:, i],
                    B[:, i],
                    format="csc",
                )
            )
        C = scipy.sparse.hstack(C, format="csc", dtype=np.int8)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        D = scipy.linalg.khatri_rao(A.toarray(), B.toarray())
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        E = khatri_rao_from_Amat(A, B)
        t5 = time.perf_counter()

        print(f"time of scipy.sparse.kron: {t1-t0}")
        print(f"time of scipy.linalg.khatri_rao: {t3-t2}")
        print(f"time of khatri_rao_for_Amat: {t5-t4}")

        assert np.allclose(C.toarray(), D)
        assert np.allclose(C.toarray(), E.toarray())

    print("ok")


if __name__ == "__main__":
    test_khatri_rao_for_Amat()
