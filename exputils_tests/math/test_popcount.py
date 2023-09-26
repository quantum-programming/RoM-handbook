import random
from exputils.math.popcount import popcount


def test_popcount():
    for i in range(10000):
        assert popcount(i) == bin(i).count("1")

    for i in range(10000):
        j = random.randint(0, 2**63 - 1)
        assert popcount(j) == bin(j).count("1")

    print("ok")


if __name__ == "__main__":
    test_popcount()
