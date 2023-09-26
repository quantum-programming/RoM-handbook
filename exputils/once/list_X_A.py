import numpy as np
from itertools import permutations
from functools import lru_cache
from tqdm import tqdm

# stabilizer_simulator 直下での実行を想定
# 実行にかなり時間がかかるの注意

# TODO: 高速化 分枝限定法？


def canonical_hash_X(X):
    """It compresses a boolean matrix X into an unsigned integer in [0, 2^64).
    Canonical hash is preserved under the application of the same permutation to rows and columns.
    """
    k = X.shape[0]
    assert X.shape == (k, k)
    assert k <= 8
    X_as_uint = X.astype(np.uint64)
    min_val = np.iinfo(np.uint64).max
    for perm_vec in permutations(range(k)):
        row_compressor = np.array([1 << p for p in perm_vec], dtype=np.uint64)
        col_compressor = np.array([1 << (p * k) for p in perm_vec], dtype=np.uint64)
        X_compressed = np.dot(row_compressor @ X_as_uint, col_compressor)
        if min_val > X_compressed:
            min_val = X_compressed
    return min_val


@lru_cache()
def list_X(k):
    assert k >= 0
    if k == 0:
        return [np.zeros((0, 0), dtype=np.int_)]
    small_Xs = list_X(k - 1)
    Xs = []
    pushed_canonical_hash = set()
    for small_X in tqdm(small_Xs):
        X_padded = np.zeros((k, k), dtype=np.int_)
        X_padded[: k - 1, : k - 1] = small_X
        for s in range(1 << k):
            X = X_padded.copy()
            for i in range(k):
                if s & (1 << i):
                    X[i][k - 1] = X[k - 1][i] = 1
            canonical_hash = canonical_hash_X(X)
            if canonical_hash not in pushed_canonical_hash:
                pushed_canonical_hash.add(canonical_hash)
                Xs.append(X)
    return np.array(Xs, dtype=np.int_)


def canonical_hash_X_A(X, A):
    k = X.shape[0]
    l = A.shape[1]
    assert X.shape == (k, k)
    assert A.shape == (k, l)
    assert l >= 1
    A_col_compressor = np.array(
        [1 << (p * k) for p in reversed(range(l))], dtype=np.uint64
    )
    X_as_uint = X.astype(np.uint64)
    A_as_uint = A.astype(np.uint64)
    min_val = (np.iinfo(np.uint64).max, np.iinfo(np.uint64).max)
    for perm_vec in permutations(range(k)):
        row_compressor = np.array([1 << p for p in perm_vec], dtype=np.uint64)
        col_compressor = np.array([1 << (p * k) for p in perm_vec], dtype=np.uint64)
        X_compressed = np.dot(row_compressor @ X_as_uint, col_compressor)
        A_vec = np.sort(row_compressor @ A_as_uint)
        A_compressed = np.dot(A_vec, A_col_compressor)
        if min_val > (X_compressed, A_compressed):
            min_val = (X_compressed, A_compressed)
    return min_val


@lru_cache()
def list_X_A(n, k):
    assert 0 <= k <= n
    if k == n:
        return list_X(k)
    small_XAs = list_X_A(n - 1, k)
    XAs = []
    pushed_canonical_hash = set()
    for small_XA in tqdm(small_XAs):
        XA_padded = np.hstack((small_XA, np.zeros((k, 1), dtype=np.int_)))
        for s in range(1 << k):
            XA = XA_padded.copy()
            for i in range(k):
                if s & (1 << i):
                    XA[i][n - 1] = 1
            canonical_hash = canonical_hash_X_A(XA[:, :k], XA[:, k:])
            if canonical_hash not in pushed_canonical_hash:
                pushed_canonical_hash.add(canonical_hash)
                XAs.append(XA)
    return np.array(XAs, dtype=np.int_)


def main():
    for n in range(1, 6 + 1):
        save_file = f"data/permdata/X_A_{n}.npz"
        print(save_file)
        save_dict = {str(k): list_X_A(n, k) for k in range(n + 1)}
        with open(save_file, "wb") as f:
            np.savez(f, **save_dict)


if __name__ == "__main__":
    main()
