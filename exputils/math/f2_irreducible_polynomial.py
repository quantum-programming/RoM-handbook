from typing import Dict, List

# We can correspond an F_2[x] element to an integer by interpreting the polynomial as a Z[x] element and substituting 2 for x.
# e.g.
# 1 + x + x^3 -> 1 + 2 + 2^3 = 11 = 0b1011


def enumerate_irreducible_polynomials(max_degree: int) -> Dict[int, List[int]]:
    end = 1 << (max_degree + 1)
    reducible = [False] * end
    for f in range(2, end // 2):
        multiples = []
        while f < end:
            multiples.append(f)
            f <<= 1
        cnt = len(multiples)
        for s in range(2, 1 << cnt):
            prod = 0
            for i in range(cnt):
                if s & (1 << i):
                    prod ^= multiples[i]
            if prod < end:
                reducible[prod] = True
    ret = dict()
    for i in range(1, end):
        if not reducible[i]:
            deg = i.bit_length() - 1
            if deg not in ret:
                ret[deg] = []
            ret[deg].append(i)
    return ret
