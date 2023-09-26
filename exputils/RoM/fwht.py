import copy

import numpy as np

from exputils.cover_states import make_cover_info
from exputils.math.fwht import FWHT


def calculate_RoM_FWHT(n_qubit: int, rho_vec: np.ndarray):
    """calculate RoM using FWHT

    Args:
        n_qubit (int): number of qubits
        rho_vec (np.ndarray): the state to decompose.

    Returns:
        RoM, coeffs, cover_generators

    Examples:
        >>> n_qubit = 2
        >>> rho_vec = make_random_quantum_state("pure", n_qubit, 0)
        >>> RoM, coeffs, cover_generators = calculate_RoM_FWHT(n_qubit, rho_vec)
        >>> print(f"{RoM=}", f"{coeffs=}", f"{cover_generators=}", sep="\n")
        >>> Amat = generators_to_Amat(n_qubit, cover_generators)
        >>> print("Amat:", *Amat.toarray().tolist(), sep="\n")

        RoM=1.6011353085560032
        coeffs=array([-0.10037725,  0.02290097,  0.05839538,  0.2190809 ,  0.03102633,
                       0.24674342, -0.03971197, -0.03805778, -0.02056102,  0.06840277,
                      -0.01195209,  0.16411034, -0.0323947 , -0.03172847,  0.04206086,
                       0.22206231, -0.00224654, -0.02353783,  0.13627274,  0.08951163])
        cover_generators=[['ZI', 'IZ'], ['XI', 'IX'], ['YI', 'IY'], ['XZ', 'ZY'], ['YZ', 'ZX']]

        Amat:
            ZI,IZ         XI,IX         YI,IY         XZ,ZY         YZ,ZX
        II [1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1]
        IX [0, 0, 0, 0,   1, 1,-1,-1,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]
        IY [0, 0, 0, 0,   0, 0, 0, 0,   1, 1,-1,-1,   0, 0, 0, 0,   0, 0, 0, 0]
        IZ [1, 1,-1,-1,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]
        XI [0, 0, 0, 0,   1,-1, 1,-1,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]
        XX [0, 0, 0, 0,   1,-1,-1, 1,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]
        XY [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   -1,1, 1,-1]
        XZ [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   1,-1, 1,-1,   0, 0, 0, 0]
        YI [0, 0, 0, 0,   0, 0, 0, 0,   1,-1, 1,-1,   0, 0, 0, 0,   0, 0, 0, 0]
        YX [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   -1,1, 1,-1,   0, 0, 0, 0]
        YY [0, 0, 0, 0,   0, 0, 0, 0,   1,-1,-1, 1,   0, 0, 0, 0,   0, 0, 0, 0]
        YZ [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   1,-1, 1,-1]
        ZI [1,-1, 1,-1,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]
        ZX [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   1, 1,-1,-1]
        ZY [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   1, 1,-1,-1,   0, 0, 0, 0]
        ZZ [1,-1,-1, 1,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]
    """
    assert rho_vec.shape[0] == 4**n_qubit
    cover_generators, cover_idxs, cover_vals = copy.deepcopy(make_cover_info(n_qubit))
    coeffs = np.array(
        [
            FWHT(n_qubit, rho_vec[cover_idxs[i]] * cover_vals[i])
            for i in range(len(cover_generators))
        ]
    )
    coeffs = np.array(coeffs).reshape(-1)
    RoM = np.sum(np.abs(coeffs))
    return RoM, coeffs, cover_generators
