from numba import njit
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm

inv_sqrt_2 = [1 / pow(2, i / 2) for i in range(20)]


@njit(cache=True)
def _subroutine(i: int):
    k = 0
    XY = 0
    idx = 0
    while i > 0:
        d = i % 3
        i //= 3
        if d != 0:
            XY += 1
        idx += d << (2 * k)
        k += 1
    return XY, idx


def make_canonical_magic_state_in_pauli_basis(t: int) -> coo_matrix:
    vals = [0 for _ in range(3**t)]
    idxs = [0 for _ in range(3**t)]
    for i in tqdm(
        range(3**t),
        disable=(t <= 15),
        desc=f"make_canonical_magic_state_in_pauli_basis({t})",
        leave=False,
    ):
        XY, idx = _subroutine(i)
        vals[i] = inv_sqrt_2[XY]
        idxs[i] = idx
    return coo_matrix((vals, ([0] * len(idxs), idxs)), shape=(1, 4**t))


def main():
    import numpy as np
    from qulacs import QuantumCircuit, QuantumState

    from exputils.state.state_in_pauli_basis import state_in_pauli_basis

    def make_canonical_magic_state(n_qubit: int, in_pauli_basis: bool) -> np.ndarray:
        state = QuantumState(n_qubit)
        circuit = QuantumCircuit(n_qubit)
        for i in range(n_qubit):
            circuit.add_H_gate(index=i)
            circuit.add_T_gate(index=i)
        circuit.update_quantum_state(state)

        v = state.get_vector()  # This is (|0>+e^{iπ/4}|1>)/√2, as stated in the paper

        if in_pauli_basis:
            # In the paper, this vector is expressed as 'ρ'
            return np.round(state_in_pauli_basis(v), decimals=10)
        else:
            return v

    for t in range(1, 6 + 1):
        arr1 = make_canonical_magic_state_in_pauli_basis(t).toarray()[0]
        arr2 = make_canonical_magic_state(t, True)
        assert np.allclose(arr1, arr2), (arr1, arr2)
        print(f"{t} ok!")


if __name__ == "__main__":
    main()
