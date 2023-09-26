from typing import List, Dict
from exputils.math.f2_irreducible_polynomial import enumerate_irreducible_polynomials


def multiply_polynomials(f: int, g: int) -> int:
    if f > g:
        f, g = g, f
    ret = 0
    while f:
        ret ^= g * (f & 1)
        f >>= 1
        g <<= 1
    return ret


def enumerate_irreducible_polynomials_naive(max_degree: int) -> Dict[int, List[int]]:
    end = 1 << (max_degree + 1)
    reducible = [False] * end
    for f in range(2, end):
        for g in range(2, end):
            h = multiply_polynomials(f, g)
            if h < end:
                reducible[h] = True
    ret = dict()
    for i in range(1, end):
        if not reducible[i]:
            deg = i.bit_length() - 1
            if deg not in ret:
                ret[deg] = []
            ret[deg].append(i)
    return ret


def test_f2_irreducible_polynomial():
    assert multiply_polynomials(0b1, 0b1) == 0b1
    assert multiply_polynomials(0b0, 0b1) == 0b0
    assert multiply_polynomials(0b1001, 0b111) == 0b111111
    assert multiply_polynomials(0b101, 0b111) == 0b11011
    assert multiply_polynomials(0b11, 0b1011) == 0b11101

    for d in range(11):
        expected = enumerate_irreducible_polynomials_naive(d)
        actual = enumerate_irreducible_polynomials(d)
        assert expected == actual

    print("ok")


if __name__ == "__main__":
    test_f2_irreducible_polynomial()
