# import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def make_custom_cmap():
    c1 = "FF5785"
    c2 = "64FF10"

    colors = [
        (lambda c: tuple(int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)))(c)
        for c in [c1, c2]
    ]

    cmap = LinearSegmentedColormap.from_list("Custom", colors, N=256)
    return cmap

    # cmap = plt.cm.get_cmap("bwr")
    # return cmap
