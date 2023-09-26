from exputils.actual_Amat import get_actual_Amat
from exputils.random_Amat import make_random_Amat
from exputils.stabilizer_group import total_stabilizer_group_size

for n_qubit in range(1, 8 + 1):
    random_Amat = make_random_Amat(n_qubit, 5).toarray()
    print(f"{random_Amat=}")

for method in ["actual", "dot_data", "gen"]:
    for n_qubit in range(1, 3 + 1):
        Amat = get_actual_Amat(n_qubit).toarray()
        random_Amat = make_random_Amat(
            n_qubit, total_stabilizer_group_size(n_qubit)
        ).toarray()

        assert all((col in Amat) for col in random_Amat)
        print(f"{n_qubit} ok!")
    print(f"{method} ok!")
print("all ok!")
