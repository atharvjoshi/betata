""" """

from pathlib import Path

from betata import plt
import pandas as pd
import numpy as np
import matplotlib.ticker as tck
from uncertainties import unumpy

TRACE_COLOR = "#762A83"


def find_header_row(filepath, header="[Data]"):
    """ """
    header_row_num = None
    with open(filepath) as file:
        for idx, line in enumerate(file.readlines(), start=1):
            if line.startswith(header):
                header_row_num = idx
                break
    return header_row_num


def extract_data(filepath, skip_header=None, usecols=None, names=True):
    """ """
    data = np.genfromtxt(
        filepath,
        delimiter=",",
        skip_header=skip_header,
        usecols=usecols,
        names=names,
    )
    return pd.DataFrame(data)


def plot_data(x, y, yerr, figsize=(6, 6)):
    """ """

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Resistivity ($\mathrm{\mu \Omega}$.cm)")

    ax.errorbar(x, y, yerr=yerr, ls="--", c=TRACE_COLOR, marker="o")

    ax.xaxis.set_major_locator(tck.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(tck.MultipleLocator(0.1))

    ax.yaxis.set_major_locator(tck.MultipleLocator(50))
    ax.yaxis.set_minor_locator(tck.MultipleLocator(10))

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    """ """

    # hall bar dimensions in m
    film_thickness = 240e-9
    channel_length = 370e-6
    channel_width = 25e-6
    x_section_area = film_thickness * channel_width

    # we combine data from the full range (fr) and low temp (lt) scans
    datafolder = Path(__file__).parents[3] / "data/fig1"
    filepath_fr = datafolder / "PPMS_ch1_130_c2_230_fullrange.dat"
    filepath_lt = datafolder / "PPMS_ch1_130_c2_230_lowtemp.dat"

    colmap = {
        3: "temperature",
        20: "resistance",
        15: "resistance_std",
    }

    # extract full range data
    skip_header_fr = find_header_row(filepath_fr)
    data_fr = extract_data(
        filepath_fr,
        skip_header=skip_header_fr + 1,  # + 1 to handle NaN values for custom names
        usecols=colmap.keys(),
        names=colmap.values(),
    )

    # extract low temp data
    skip_header_lt = find_header_row(filepath_lt)
    data_lt = extract_data(
        filepath_lt,
        skip_header=skip_header_lt + 1,  # + 1 to handle NaN values for custom names
        usecols=colmap.keys(),
        names=colmap.values(),
    )

    # combine both datasets and trim temperature domain to below 1.5K
    data = pd.concat([data_fr[data_fr["temperature"] < 1.5], data_lt])
    data = data.sort_values(by="temperature")

    # convert resistance to resistivity
    resistance_uarr = unumpy.uarray(data["resistance"], data["resistance_std"])
    resistivity_uarr = resistance_uarr * x_section_area / channel_length
    # change resistivity units to microohm.cm
    resistivity_uarr *= 1e8

    data["resistivity"] = unumpy.nominal_values(resistivity_uarr)
    data["resistivity_std"] = unumpy.std_devs(resistivity_uarr)

    figure = plot_data(
        data["temperature"],
        data["resistivity"],
        data["resistivity_std"],
        figsize=(6, 5),
    )

    figsavepath = Path(__file__).parents[3] / "out/fig1/PPMS.png"

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")

    plt.show()
