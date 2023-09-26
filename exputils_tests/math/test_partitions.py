from typing import List, Union

from exputils.math.partitions import partitions

# Reference: https://oeis.org/A000041
partition_count = [
    1,
    1,
    2,
    3,
    5,
    7,
    11,
    15,
    22,
    30,
    42,
    56,
    77,
    101,
    135,
    176,
    231,
    297,
    385,
    490,
    627,
    792,
    1002,
    1255,
    1575,
    1958,
    2436,
    3010,
    3718,
    4565,
    5604,
    6842,
    8349,
    10143,
    12310,
    14883,
    17977,
    21637,
    26015,
    31185,
    37338,
    44583,
    53174,
    63261,
    75175,
    89134,
    105558,
    124754,
    147273,
    173525,
]

# It excludes n = 0.
for n in range(1, len(partition_count)):
    assert len(partitions(n)) == partition_count[n]
    assert partitions(n) == sorted(partitions(n))

for n in range(10):
    partition_result = partitions(n)
    for lower_bound in range(1, n + 1):
        for upper_bound in range(1, n + 1):
            filtered = [
                partition
                for partition in partition_result
                if lower_bound <= min(partition) and max(partition) <= upper_bound
            ]
            assert filtered == partitions(n, lower_bound, upper_bound)

print("ok")
