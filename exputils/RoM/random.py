from typing import Tuple, Union

import numpy as np

from exputils.random_Amat import make_random_Amat
from exputils.RoM.custom import calculate_RoM_custom
from exputils.stabilizer_group import total_stabilizer_group_size


def calculate_RoM_random(
    n_qubit: int,
    rho_vec: np.ndarray,
    K: Union[int, float],
    verbose: bool = False,
    method: str = "scipy",
    return_dual: bool = False,
    crossover: bool = True,
    presolve: bool = False,
) -> Union[Tuple[float, np.ndarray], Tuple[float, np.ndarray, np.ndarray]]:
    if K <= 1:
        assert isinstance(K, float)
        sz = int(K * total_stabilizer_group_size(n_qubit))
    else:
        assert isinstance(K, int)
        sz = K
    Amat = make_random_Amat(n_qubit, sz)
    return calculate_RoM_custom(
        Amat,
        rho_vec,
        verbose=verbose,
        method=method,
        return_dual=return_dual,
        crossover=crossover,
        presolve=presolve,
    )
