import os
import subprocess
from glob import glob

import numpy as np


def convert_g6_to_npy():
    graph_dir = os.path.join(os.path.dirname(__file__), "../../data/LCgraphs/")
    g6_files = glob(graph_dir + "*.g6")
    for file in g6_files:
        print("converting:", file)
        with open(graph_dir + "_tmp_convert.txt", "w") as tmp:
            subprocess.run(
                [
                    graph_dir + "a.out",
                    "-a",
                    "-q",
                    file,
                ],
                stdout=tmp,
            )
        with open(graph_dir + "_tmp_convert.txt", "r") as tmp:
            lines = tmp.readlines()
        n = int(lines[0])
        graph_count = len(lines) // (n + 1)
        graphs = np.empty((graph_count, n), dtype=np.int_)
        for i in range(graph_count):
            graph = []
            for index in range(i * (n + 1) + 1, (i + 1) * (n + 1)):
                graph.append(int(lines[index].strip()[::-1], 2))
            graphs[i] = graph
        with open(file.replace(".g6", ".npy"), "wb") as f:
            np.save(f, graphs)
        os.remove(graph_dir + "_tmp_convert.txt")


def main():
    convert_g6_to_npy()


if __name__ == "__main__":
    main()
