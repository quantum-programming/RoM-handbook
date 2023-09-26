import numpy as np

H_state = np.array([(1.0 + 0.0j) / np.sqrt(2), 0.5 + 0.5j], dtype=np.complex_)

beta = np.arccos(1 / np.sqrt(3))
F_state = np.array(
    [np.cos(beta / 2), np.exp(1j * np.pi / 4) * np.sin(beta / 2)],
    dtype=np.complex_,
)

I = np.array([[1, 0], [0, 1]], dtype=np.complex_)
X = np.array([[0, 1], [1, 0]], dtype=np.complex_)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex_)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex_)

print(H_state.T.conj() @ H_state)
print(np.outer(H_state, H_state.T.conj()))
print(1 / 2 * (I + (1 / np.sqrt(2)) * (X + Y)))
assert np.allclose(
    np.outer(H_state, H_state.T.conj()), 1 / 2 * (I + (1 / np.sqrt(2)) * (X + Y))
)

print(F_state.T.conj() @ F_state)
print(np.outer(F_state, F_state.T.conj()))
print(1 / 2 * (I + (1 / np.sqrt(3)) * (X + Y + Z)))
assert np.allclose(
    np.outer(F_state, F_state.T.conj()), 1 / 2 * (I + (1 / np.sqrt(3)) * (X + Y + Z))
)
