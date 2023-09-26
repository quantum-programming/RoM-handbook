from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def pauli_dot(A: Tuple[int, str], B: Tuple[int, str]) -> Tuple[int, str]:
    """dot product of two Pauli strings

    Args:
        A (Tuple[int, str]): (phase(e^{kiπ/2}), Pauli string)
        B (Tuple[int, str]): (phase(e^{kiπ/2}), Pauli string)

    Returns:
        Tuple[int, str]: (phase(e^{kiπ/2}), Pauli string)

    Examples:
        (+1)X * (-1)Y = (-i)Z
        >>> pauli_dot((0, "X"), (2, "Y")) == (3, "Z")

        (+i)ZX * (-i)ZY = (+i)IZ
        >>> pauli_dot((1, "ZX"), (3, "ZY")) == (1, "IZ")

        (+i)IXY * (+1)ZIX = (+1)ZXZ
        >>> pauli_dot((1, "IXY"), (0, "ZIX")) == (0, "ZXZ")
    """
    chars = [
        ["I", "X", "Y", "Z"],
        ["X", "I", "Z", "Y"],
        ["Y", "Z", "I", "X"],
        ["Z", "Y", "X", "I"],
    ]
    phases = [
        [0, 0, 0, 0],
        [0, 0, +1, -1],
        [0, -1, 0, +1],
        [0, +1, -1, 0],
    ]
    phase = A[0] + B[0]
    s = ""
    for a, b in zip(A[1], B[1]):
        a_idx = "IXYZ".find(a)
        b_idx = "IXYZ".find(b)
        assert a_idx != -1
        assert b_idx != -1
        s += chars[a_idx][b_idx]
        phase += phases[a_idx][b_idx]
    return (phase % 4, s)


@njit(cache=True)
def pauli_dot_without_str(
    A: Tuple[int, np.ndarray], B: Tuple[int, np.ndarray]
) -> Tuple[int, np.ndarray]:
    """quaternary representation of Pauli string in np.ndarray

    Examples:
        (+1)X * (-1)Y = (-i)Z
        >>> pauli_dot_without_str((0, np.array([1])), (2, np.array([2]))) == (3, np.array([3]))

        (+i)ZX * (-i)ZY = (+i)IZ
        >>> pauli_dot_without_str((1, np.array([3, 1])), (3, np.array([3, 2]))) == (1, np.array([0, 3]))

        (+i)IXY * (+1)ZIX = (+1)ZXZ
        >>> pauli_dot_without_str((1, np.array([0, 1, 2])), (0, np.array([3, 0, 1]))) == (0, np.array([3, 1, 3]))
    """
    chars = np.array(
        [0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0],
        dtype=np.int8,
    )
    phases = np.array(
        [0, 0, 0, 0, 0, 0, +1, -1, 0, -1, 0, +1, 0, +1, -1, 0],
        dtype=np.int32,
    )
    c = A[1] * 4 + B[1]
    phase = np.sum(phases[c])
    return (A[0] + B[0] + phase) % 4, chars[c]


@njit(cache=True)
def pauli_dot_without_sign(A: int, B: int) -> int:
    """quaternary representation of Pauli string in int

    Examples:
        (+1)X * (-1)Y = (-i)Z
        >>> pauli_dot_without_sign(1, 2) == 3

        (+i)ZX * (-i)ZY = (+i)IZ
        >>> pauli_dot_without_sign(3+1*4, 3+2*4) == 0+3*4

        (+i)IXY * (+1)ZIX = (+1)ZXZ
        >>> pauli_dot_without_sign(0+1*4+2*4**2, 3+0*4+1*4**2) == 3+1*4+3*4**2
    """
    chars = np.array(
        [0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0],
        dtype=np.int8,
    )
    ret = 0
    digit = 0
    while A > 0 or B > 0:
        ret += chars[(A % 4) * 4 + (B % 4)] * (4**digit)
        digit += 1
        A //= 4
        B //= 4
    return ret


if __name__ == "__main__":
    print(pauli_dot((0, "X"), (2, "Y")))
    print(pauli_dot_without_str((0, np.array([1])), (2, np.array([2]))))
    print(f"{pauli_dot_without_sign(1, 2):0b}")

    print(pauli_dot((1, "ZX"), (3, "ZY")))
    print(pauli_dot_without_str((1, np.array([3, 1])), (3, np.array([3, 2]))))
    print(f"{pauli_dot_without_sign(3+1*4, 3+2*4):0b}")

    print(pauli_dot((1, "IXY"), (0, "ZIX")))
    print(pauli_dot_without_str((1, np.array([0, 1, 2])), (0, np.array([3, 0, 1]))))
    print(f"{pauli_dot_without_sign(0+1*4+2*4**2, 3+0*4+1*4**2):0b}")
