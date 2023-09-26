import random
import warnings
from typing import Tuple

import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm

from exputils.divide import (
    divide_to_parts,
    new_idxs_by_division,
    rev_new_idxs_by_division,
)
from exputils.dot.dot_product import compute_all_dot_products
from exputils.dot.load_data import load_dot_data
from exputils.math.fwht import sylvesters
from exputils.math.khatri_rao import khatri_rao_from_data_idxs
from exputils.stabilizer_group import total_stabilizer_group_size


def _fast_argsort(vals: np.ndarray, K: int):
    """fast argsort

    Warning: This function is not stable. The result might be smaller than K.

    This function passes the test below.

    >>> topK_idxs_uv, botK_idxs_uv = _fast_argsort(vals, K)
    >>> answer_top = np.argpartition(-vals, K)[:K]
    >>> answer_bot = np.argpartition(vals, K)[:K]
    >>> assert set(topK_idxs_uv.tolist()).issubset(answer_top.tolist())
    >>> assert set(botK_idxs_uv.tolist()).issubset(answer_bot.tolist())
    """
    if vals.size < 10000:
        topK_idxs_uv = np.argsort(-vals)[:K]
        botK_idxs_uv = np.argsort(vals)[:K]
    else:
        ma, mi = np.max(vals), np.min(vals)

        top_args_temp = np.where(vals > ma - (ma - mi) / 5)[0]
        K1 = min(K, len(top_args_temp) - 1)
        # if K1 < K:
        #     warnings.warn(f"fast argsort: K1({K1}) < K({K})")
        topK_idxs_uv = top_args_temp[np.argpartition(-vals[top_args_temp], K1)[:K1]]

        bot_args_temp = np.where(vals < mi + (ma - mi) / 5)[0]
        K2 = min(K, len(bot_args_temp) - 1)
        # if K2 < K:
        #     warnings.warn(f"fast argsort: K2({K2}) < K({K})")

        botK_idxs_uv = bot_args_temp[np.argpartition(vals[bot_args_temp], K2)[:K2]]

    return topK_idxs_uv, botK_idxs_uv


def _make_Amat(idxs, n, _d, _r):
    H = sylvesters(n)
    q, r = np.divmod(idxs, 2**n)
    return (_d[q] * H[r]), _r[q]


def _routine_compute_topK_tensor_type_dot_products(
    n1: int,
    n2: int,
    rho_vec: np.ndarray,
    K: int,
    d1: np.ndarray,
    r1: np.ndarray,
    d2: np.ndarray,
    r2: np.ndarray,
):
    assert 1 < K
    assert rho_vec.shape == (4 ** (n1 + n2),)
    _U, S, _V = np.linalg.svd(rho_vec.reshape(4**n1, 4**n2), full_matrices=False)
    if np.isclose(S[1], 0.0):
        U = np.array([_U[:, 0] * np.sqrt(S[0])])
        V = np.array([_V[0, :] * np.sqrt(S[0])])
        S = np.array([S[0]])
    else:
        U = _U
        V = _V
        for i in range(len(S)):
            U[:, i] *= np.sqrt(S[i])
            V[i, :] *= np.sqrt(S[i])
        U = U.T
    assert U.ndim == V.ndim == 2
    assert U.shape[0] == V.shape[0] == (1 if len(S) == 1 else 4 ** min(n1, n2))
    assert U.shape[1] == 4**n1
    assert V.shape[1] == 4**n2
    assert np.allclose(U.T @ V, rho_vec.reshape(4**n1, 4**n2))

    PARAM_SIZE = 1e10

    orig_idxs_u = np.array(
        random.sample(
            range(total_stabilizer_group_size(n1) // (2**n1)),
            min(
                total_stabilizer_group_size(n1) // (2**n1),
                int(np.sqrt(PARAM_SIZE // ((2**n1) * U.shape[0]))),
            ),
        ),
        dtype=np.int32,
    )
    orig_idxs_v = np.array(
        random.sample(
            range(total_stabilizer_group_size(n2) // (2**n2)),
            min(
                total_stabilizer_group_size(n2) // (2**n2),
                int(np.sqrt(PARAM_SIZE // ((2**n2) * V.shape[0]))),
            ),
        ),
        dtype=np.int32,
    )

    _d1 = d1[orig_idxs_u]
    _r1 = r1[orig_idxs_u]
    U_dot = np.vstack(
        [compute_all_dot_products(n1, U[i], _d1, _r1) for i in range(len(S))]
    )
    _d2 = d2[orig_idxs_v]
    _r2 = r2[orig_idxs_v]
    V_dot = np.vstack(
        [compute_all_dot_products(n2, V[i], _d2, _r2) for i in range(len(S))]
    )
    dots = U_dot.T @ V_dot

    raveled_dots = dots.ravel()
    topK_idxs_uv, botK_idxs_uv = _fast_argsort(raveled_dots, K)
    topK_idxs_u, topK_idxs_v = np.unravel_index(topK_idxs_uv, dots.shape)
    botK_idxs_u, botK_idxs_v = np.unravel_index(botK_idxs_uv, dots.shape)

    d1, r1, c1 = khatri_rao_from_data_idxs(
        *_make_Amat(topK_idxs_u, n1, _d1, _r1),
        *_make_Amat(topK_idxs_v, n2, _d2, _r2),
    )
    d2, r2, c2 = khatri_rao_from_data_idxs(
        *_make_Amat(botK_idxs_u, n1, _d1, _r1),
        *_make_Amat(botK_idxs_v, n2, _d2, _r2),
    )

    Amat = csc_matrix(
        (
            np.concatenate([d1, d2]),
            (np.concatenate([r1, r2]), np.concatenate([c1, c2 + len(topK_idxs_u)])),
        ),
        shape=(4 ** (n1 + n2), len(topK_idxs_u) + len(botK_idxs_u)),
        dtype=np.int8,
    )

    ret_dots = np.concatenate([raveled_dots[topK_idxs_uv], raveled_dots[botK_idxs_uv]])

    return Amat, ret_dots


def compute_topK_tensor_type_dot_products(
    n: int, rho_vec: np.ndarray, K: int
) -> Tuple[csc_matrix, np.ndarray]:
    Amat_list = []
    dots_list = []

    for n1 in range(1, n // 2 + 1):
        n2 = n - n1
        if n2 > 6:
            continue
        d1, r1 = load_dot_data(n1)
        d2, r2 = load_dot_data(n2)
        for div in tqdm(
            divide_to_parts(n, (n1, n2)),
            desc=f"topK tensor dot(n1,n2:{n1},{n2})",
            leave=False,
        ):
            new_idxs = new_idxs_by_division(div)
            rev_idxs = rev_new_idxs_by_division(new_idxs)

            Amat, dots = _routine_compute_topK_tensor_type_dot_products(
                n1, n2, rho_vec[new_idxs], K, d1, r1, d2, r2
            )
            dots = np.concatenate([dots, np.full(2 * K - len(dots), np.nan)])

            Amat_list.append(Amat[rev_idxs])
            dots_list.append(dots)

    raveled_dots = np.concatenate(dots_list, axis=0)
    topK_idxs_uv = np.argpartition(-raveled_dots, K)[:K]
    assert np.all(np.isfinite(raveled_dots[topK_idxs_uv]))
    topK_idxs_u, topK_idxs_v = np.unravel_index(topK_idxs_uv, (len(dots_list), 2 * K))
    topK_Amat_list_idxs = [[] for _ in range(len(Amat_list))]
    for u, v in zip(topK_idxs_u.tolist(), topK_idxs_v.tolist()):
        topK_Amat_list_idxs[u].append(v)
    botK_idxs_uv = np.argpartition(raveled_dots, K)[:K]
    assert np.all(np.isfinite(raveled_dots[botK_idxs_uv]))
    botK_idxs_u, botK_idxs_v = np.unravel_index(botK_idxs_uv, (len(dots_list), 2 * K))
    botK_Amat_list_idxs = [[] for _ in range(len(Amat_list))]
    for u, v in zip(botK_idxs_u.tolist(), botK_idxs_v.tolist()):
        botK_Amat_list_idxs[u].append(v)

    Amat = scipy.sparse.hstack(
        [A[:, topIdx] for A, topIdx in zip(Amat_list, topK_Amat_list_idxs)]
        + [A[:, botIdx] for A, botIdx in zip(Amat_list, botK_Amat_list_idxs)]
    )

    dots = rho_vec.T @ Amat
    unique_idxs = np.unique(dots, return_index=True)[1]
    dots = dots[unique_idxs]
    Amat = Amat[:, unique_idxs]

    return Amat, dots
