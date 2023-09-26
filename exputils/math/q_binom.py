from functools import reduce


def q_factorial(k: int):
    """q_factorial. (q is fixed to 2)
    [k]_q! = \prod_{i=1}^k (1 + q + q^2 + ... + q^{i-1})

    Reference:
        https://mathworld.wolfram.com/q-Factorial.html
    """
    assert 0 <= k
    ret = 1
    for i in range(1, k + 1):
        ret *= sum([2**j for j in range(i)])
    if k <= 5:
        assert ret == [1, 1, 3, 21, 315, 9765][k]
    return ret


def q_binomial(n: int, k: int):
    """q_binomial. (q is fixed to 2)
    [n k]_q = \frac{[n]_q!}{[k]_q! [n-k]_q!}

    Reference:
        https://mathworld.wolfram.com/q-BinomialCoefficient.html
    """
    ret1 = q_factorial(n) // (q_factorial(k) * q_factorial(n - k))
    ret2 = reduce(
        lambda x, i: x * (1 - (2 ** (n - i))) / (1 - (2 ** (i + 1))),
        range(k),
        1,
    )
    assert ret1 == int(ret2), f"{ret1=} {ret2=}"
    if k == 0 or n == k:
        assert ret1 == 1
    elif n == 2 and k == 1:
        assert ret1 == 3
    elif (n == 3 and k == 1) or (n == 3 and k == 2):
        assert ret1 == 7
    elif (n == 4 and k == 1) or (n == 4 and k == 3):
        assert ret1 == 15
    elif n == 4 and k == 2:
        assert ret1 == 35
    return ret1


def main():
    for n in range(1, 10 + 1):
        print(f"{n=}")
        print(q_factorial(n))
        for k in range(n + 1):
            print(q_binomial(n, k), end=" ")
        print()


if __name__ == "__main__":
    main()
