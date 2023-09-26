from typing import Tuple, Union

import numpy as np

from exputils.actual_Amat import get_actual_Amat
from exputils.RoM.custom import calculate_RoM_custom


def calculate_RoM_actual(
    n_qubit: int,
    rho_vec: np.ndarray,
    verbose: bool = False,
    method: str = "scipy",
    return_dual: bool = False,
    crossover: bool = True,
    presolve: bool = False,
) -> Union[Tuple[float, np.ndarray], Tuple[float, np.ndarray, np.ndarray]]:
    return calculate_RoM_custom(
        custom_Amat=get_actual_Amat(n_qubit),
        rho_vec=rho_vec,
        verbose=verbose,
        method=method,
        return_dual=return_dual,
        crossover=crossover,
        presolve=presolve,
    )
