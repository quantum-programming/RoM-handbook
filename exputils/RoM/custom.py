import warnings
from typing import Tuple, Union

import numpy as np
import scipy.sparse
from scipy.optimize import linprog
from scipy.sparse import csc_matrix


def calculate_RoM_custom(
    custom_Amat: Union[np.ndarray, csc_matrix],
    rho_vec: np.ndarray,
    verbose: bool = False,
    method: str = "scipy",
    return_dual: bool = False,
    crossover: bool = True,
    presolve: bool = False,
) -> Union[Tuple[float, np.ndarray], Tuple[float, np.ndarray, np.ndarray]]:
    """calculate 'Robustness of Magic' with customized stabilizer states.

    This function solves the following l1-norm minimization problem.

    minimize_{x} ||x||_1
    s.t. custom_Amat @ x = rho_vec

    This problem is equivalent to the following linear programming problem.

    minimize_{u} u.sum()
    s.t. (custom_A, -custom_A) @ u = rho_vec
          u >= 0

    Args:
        custom_Amat (Union[np.ndarray, csc_matrix]): Matrix of customized stabilizer states.
        rho_vec (np.ndarray): the state to decompose.
        verbose (bool, optional): If verbose is True, report progress. Defaults to False.
        method (str, optional): Method to calculate RoM. Defaults to "scipy".
                                When "gurobi" is given, gurobipy is required.
        return_dual (bool, optional): If return_dual is True, it returns dual variables
                                      as a third element. Defaults to False.
        crossover (bool, optional): If crossover is False, disable crossover in gurobi.
                                    Defaults to True.
        presolve (bool, optional): If presolve is True, enable presolve.
                                   Defaults to False.

    Returns:
        Union[Tuple[float, np.ndarray, Tuple[float, np.ndarray, np.ndarray]]: (RoM, coeff[, dual])
    """
    assert isinstance(rho_vec, np.ndarray)
    if rho_vec.ndim == 1:
        rho_vec_P = rho_vec
    elif rho_vec.ndim == 2:
        assert rho_vec.shape[0] == 1
        rho_vec_P = np.squeeze(rho_vec)
    else:
        assert False, rho_vec.shape
    assert (
        custom_Amat.ndim == 2
        and rho_vec_P.ndim == 1
        and custom_Amat.shape[0] == rho_vec_P.shape[0]
    ), (custom_Amat.shape, rho_vec_P.shape)
    sz = custom_Amat.shape[1]

    if method == "scipy":
        A_and_minus_A = (
            np.hstack([custom_Amat, -custom_Amat])
            if isinstance(custom_Amat, np.ndarray)
            else scipy.sparse.hstack([custom_Amat, -custom_Amat])
        )
        res = linprog(
            c=np.ones(2 * sz),
            A_eq=A_and_minus_A,
            b_eq=rho_vec_P,
            bounds=[0, None],
        )
        if not res.success:
            warnings.warn(
                f"RoM calculation failed. ({rho_vec.shape=},{custom_Amat.shape=})"
            )
            ret = (np.nan, np.nan, np.nan)
        else:
            ret = (
                res.fun,
                np.ravel(res.x)[:sz] - np.ravel(res.x)[sz:],
                res.eqlin.marginals,
            )
    elif method == "gurobi":
        import gurobipy as gp
        from gurobipy import GRB

        # https://www.gurobi.com/documentation/10.0/examples/python_examples.html
        # https://www.gurobi.com/documentation/10.0/quickstart_linux/cs_example_matrix1_py.html
        # https://www.gurobi.com/wp-content/plugins/hd_documentations/documentation/9.0/refman.pdf#page=558&zoom=100,96,408
        m = gp.Model("RoM")

        if not verbose:
            m.Params.LogToConsole = 0

        # https://www.gurobi.com/wp-content/plugins/hd_documentations/documentation/9.0/refman.pdf#page=815&zoom=100,96,665
        # https://www.msi.co.jp/solution/nuopt/glossary/term_858eb936fb0ba142e517d9dea5d922fe4463fafb.html
        # Using the dual simplex method is more memory efficient,
        # but takes more computation time.
        # >>> m.Params.Method = 1  # dual simplex method

        # https://support.gurobi.com/hc/en-us/community/posts/360043330491-The-role-of-crossover-in-linear-programming
        # https://support.gurobi.com/hc/en-us/community/posts/13739787694097-How-can-I-disable-Crossover-in-my-specific-case-
        # https://groups.google.com/g/gurobi/c/jRA8ljn0XBs
        if not crossover:
            m.Params.method = 2  # barrier method
            m.Params.crossover = 0

        # https://support.gurobi.com/hc/en-us/articles/360024738352-How-does-presolve-work-
        # https://www.gurobi.com/documentation/8.1/refman/crossover.html#parameter:Crossover
        # https://stackoverflow.com/questions/38052754/drawbacks-of-avoiding-crossover-after-barrier-solve-in-linear-program
        if not presolve:
            m.Params.Presolve = 0

        x = m.addMVar(shape=2 * sz, lb=0, obj=0.0)
        m.setObjective(np.ones(2 * sz) @ x, GRB.MINIMIZE)
        A = scipy.sparse.hstack([custom_Amat, -custom_Amat])
        m.addConstr(A @ x == rho_vec_P)
        m.optimize()
        if m.status != GRB.OPTIMAL:
            warnings.warn(
                f"RoM calculation failed. ({rho_vec.shape=},{custom_Amat.shape=})"
            )
            ret = (np.nan, np.nan, np.nan)
        else:
            ret = (m.ObjVal, x.X[:sz] - x.X[sz:], np.array(m.Pi))
    else:
        assert False, f"method must be 'scipy' or 'gurobi', but {method} is given."

    if not return_dual:
        ret = (ret[0], ret[1])
    return ret
