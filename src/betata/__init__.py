import numpy as np
import matplotlib.pyplot as plt

rcparams = plt.rcParams

rcparams["font.sans-serif"] = "Avenir"
rcparams["font.family"] = "sans-serif"
rcparams["font.size"] = 20

rcparams["lines.linewidth"] = 3
rcparams["lines.markersize"] = 10

rcparams["axes.linewidth"] = 2.0
rcparams["axes.titlesize"] = 20
rcparams["axes.labelsize"] = 20

rcparams["xtick.labelsize"] = 20
rcparams["ytick.labelsize"] = 20

rcparams["xtick.major.size"] = 10
rcparams["xtick.major.width"] = 3
rcparams["xtick.minor.size"] = 6
rcparams["xtick.minor.width"] = 2

rcparams["ytick.major.size"] = 10
rcparams["ytick.major.width"] = 3
rcparams["ytick.minor.size"] = 6
rcparams["ytick.minor.width"] = 2

rcparams["axes.spines.top"] = False
rcparams["axes.spines.right"] = False


def get_color_cycle(cmap, num, start, stop):
    """ """
    colormap = plt.get_cmap(cmap)
    return [colormap(i) for i in np.linspace(start, stop, num)]


def get_blues(num, start=0.4, stop=0.9):
    """ """
    return get_color_cycle("Blues", num, start, stop)


def get_purples(num, start=0.4, stop=0.9):
    """ """
    return get_color_cycle("Purples", num, start, stop)
