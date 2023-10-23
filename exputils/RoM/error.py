import warnings
from typing import Tuple, Union

import numpy as np
import scipy.sparse
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from exputils.dot.tensor_type_dot import compute_topK_tensor_type_dot_products
from exputils.random_Amat import make_random_Amat
from exputils.RoM.dot import get_topK_Amat
from exputils.RoM.fwht import calculate_RoM_FWHT
from exputils.stabilizer_group import total_stabilizer_group_size


def LP_with_error_term(
    custom_Amat: Union[np.ndarray, csc_matrix],
    rho_vec: np.ndarray,
    predicted_error_coeff: float,
    verbose: bool = False,
    method: str = "scipy",
    crossover: bool = True,
    presolve: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """calculate 'Robustness of Magic' with customized stabilizer states.

    This function solves the following l1-norm minimization problem.

    minimize_{x} ||x||_1 + predicted_error_coeff * ||e||_1
    s.t. custom_Amat @ x - rho_vec = e

    This problem is equivalent to the following linear programming problem.

    minimize_{u,e} u.sum() + predicted_error_coeff * e.sum()
    s.t. (custom_A, -custom_A eye -eye) @ (u,e).T = rho_vec
          u,e >= 0

    Args:
        custom_Amat (Union[np.ndarray, csc_matrix]): Matrix of customized stabilizer states.
        rho_vec (np.ndarray): the state to decompose.
        predicted_error_coeff (float): Coefficient of error term.
        verbose (bool, optional): If verbose is True, report progress. Defaults to False.
        method (str, optional): Method to calculate RoM. Defaults to "scipy".
                                When "gurobi" is given, gurobipy is required.
        crossover (bool, optional): If crossover is False, disable crossover in gurobi.
                                    Defaults to True.
        presolve (bool, optional): If presolve is True, enable presolve.
                                   Defaults to False.
    Returns:
        Tuple[float, np.ndarray, np.ndarray]: (RoM, coeff, error_coeff)
    """
    assert custom_Amat.ndim == 2
    h = custom_Amat.shape[0]
    w = custom_Amat.shape[1]
    if method == "scipy":
        A_eq = (
            np.hstack([custom_Amat, -custom_Amat, np.eye(h), -np.eye(h)])
            if isinstance(custom_Amat, np.ndarray)
            else scipy.sparse.hstack(
                [custom_Amat, -custom_Amat, scipy.sparse.eye(h), -scipy.sparse.eye(h)]
            )
        )
        res = linprog(
            c=np.hstack([np.ones(2 * w), predicted_error_coeff * np.ones(2 * h)]),
            A_eq=A_eq,
            b_eq=rho_vec,
            bounds=[0, None],
            disp=verbose,
        )
        if not res.success:
            warnings.warn(
                f"RoM calculation failed. ({rho_vec.shape=},{custom_Amat.shape=})"
            )
            return np.nan, np.nan, np.nan
        else:
            X = np.ravel(res.x)
            return (
                res.fun,
                X[:w] - X[w : 2 * w],
                X[2 * w : 2 * w + h] - X[2 * w + h : 2 * w + 2 * h],
            )
    elif method == "gurobi":
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("RoM")

        if not verbose:
            m.Params.LogToConsole = 0

        if not crossover:
            m.Params.method = 2
            m.Params.Crossover = 0

        if not presolve:
            m.Params.presolve = 0

        x = m.addMVar(shape=2 * w + 2 * h, lb=0, obj=0.0)
        obj = np.hstack([np.ones(2 * w), predicted_error_coeff * np.ones(2 * h)])
        m.setObjective(obj @ x, GRB.MINIMIZE)
        A = scipy.sparse.hstack(
            [custom_Amat, -custom_Amat, scipy.sparse.eye(h), -scipy.sparse.eye(h)]
        )
        m.addConstr(A @ x == rho_vec)
        m.optimize()
        if m.status != GRB.OPTIMAL:
            warnings.warn(
                f"RoM calculation failed. ({rho_vec.shape=},{custom_Amat.shape=})"
            )
            return np.nan, np.nan, np.nan
        else:
            return (
                m.ObjVal,
                x.X[:w] - x.X[w : 2 * w],
                x.X[2 * w : 2 * w + h] - x.X[2 * w + h : 2 * w + 2 * h],
            )
    else:
        assert False, f"method must be 'scipy' or 'gurobi', but {method} is given."


def calculate_RoM_error(
    n_qubit: int,
    rho_vec: np.ndarray,
    Amat: Union[np.ndarray, csc_matrix],
    method: str = "scipy",
    verbose: bool = False,
    crossover: bool = True,
    presolve: bool = False,
):
    if verbose:
        print("now generating Amat...")

    predicted_error_coeff = 1.0

    for phase in range(3):
        if verbose:
            print(
                "done",
                "now solving " + ["first", "second"][phase - 1] + "phase LP...",
                sep="\n",
            )

        _, coeffs, errors = LP_with_error_term(
            Amat,
            rho_vec,
            predicted_error_coeff,
            verbose=verbose,
            method=method,
            crossover=crossover,
            presolve=presolve,
        )
        Amat = Amat[:, np.abs(coeffs) > 1e-15]
        coeffs = coeffs[np.abs(coeffs) > 1e-15]
        assert np.allclose(errors, rho_vec - Amat @ coeffs), (
            errors,
            rho_vec - Amat @ coeffs,
        )
        errors = rho_vec - Amat @ coeffs

        error_RoM, error_coeffs, error_generators = calculate_RoM_FWHT(n_qubit, errors)

        RoM = np.sum(np.abs(coeffs)) + error_RoM
        predicted_error_coeff = min(10.0, error_RoM / (np.sum(np.abs(errors)) + 1e-10))
        if verbose:
            print(f"{predicted_error_coeff=}")

    if verbose:
        print("done")

    return RoM, Amat, coeffs, error_generators, error_coeffs
