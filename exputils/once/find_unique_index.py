from importlib.resources import files
import numpy as np

from exputils.perm_Amat import make_group_to_perm_unique


if __name__ == "__main__":
    for n in range(1, 7 + 1):
        before_file = files("exputils").joinpath(
            f"../data/LCgraphs/group_to_perm_{n}.npz"
        )
        after_file = files("exputils").joinpath(
            f"../data/LCgraphs/group_to_perm_{n}.npz"
        )
        loaded = np.load(before_file)
        idxs = loaded["idxs"]
        data = loaded["data"]
        used_indices, _ = make_group_to_perm_unique(n, idxs, data)
        new_idxs = idxs[used_indices]
        new_data = data[used_indices]
        _, unique_indices = make_group_to_perm_unique(n, new_idxs, new_data)
        print(f"{n}:", idxs.shape, "->", new_idxs.shape)
        np.savez_compressed(
            after_file, idxs=new_idxs, data=new_data, unique_indices=unique_indices
        )
