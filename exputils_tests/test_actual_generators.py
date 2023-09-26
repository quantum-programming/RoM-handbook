from exputils.actual_generators import get_actual_generators
from exputils.stabilizer_group import (
    stabilizer_group_size_from_gen,
    total_stabilizer_group_size,
)


def test_actual_generators():
    for n_qubit in range(1, 4 + 1):
        generators = get_actual_generators(n_qubit)
        num_of_generators = len(generators)
        assert stabilizer_group_size_from_gen(
            generators
        ) == total_stabilizer_group_size(n_qubit)
        print(
            f"{n_qubit=}, {num_of_generators=}, {generators[:min(num_of_generators,10)]=}"
        )

    # cacheが動作しているかの確認
    print(len(get_actual_generators(3)))
    print(len(get_actual_generators(2)))
    print(len(get_actual_generators(4)))
    print(len(get_actual_generators(1)))


if __name__ == "__main__":
    test_actual_generators()
