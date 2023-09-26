from matplotlib.colors import LinearSegmentedColormap


def make_custom_cmap():
    colors = [(1, 0, 0), (0, 1, 0)]
    cmap = LinearSegmentedColormap.from_list("Custom", colors, N=256)
    return cmap
