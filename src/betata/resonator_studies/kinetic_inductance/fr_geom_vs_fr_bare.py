"""
Plot simulated frequency (with no sheet inductance) `f_geom` against measured bare frequency 'fr_bare'.

Run the script "alpha_bare.py" to find and save `fr_bare` before running this script.

"""

from pathlib import Path

import numpy as np
from matplotlib import ticker

from betata import plt, get_blues
from betata.resonator_studies.resonator import Resonator, load_resonators


if __name__ == "__main__":
    """ """

    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    figsavepath = resonator_folder / "fr_sim_vs_fr_meas.png"

    resonators: list[Resonator] = load_resonators()
    data: dict[int, tuple[list, list]] = {}

    # only use CPW resonators from certain films for this subfigure, for neatness
    included_films = ["F1", "F2", "F5", "F8", "F9", "F11", "F14"]
    for resonator in resonators:
        film_name = resonator.name.split("_")[1]
        if resonator.type == "CPW" and film_name in included_films:
            thickness = resonator.film_thickness
            if thickness not in data.keys():
                data[thickness] = ([], [])

            data[resonator.film_thickness][0].append(resonator.fr_geom)
            data[resonator.film_thickness][1].append(resonator.fr_bare)

    sorted_data = dict(sorted(data.items()))

    fig, ax = plt.subplots(figsize=(5, 5))
    blues = get_blues(len(sorted_data.keys()))

    for idx, (thickness, (fr_geoms, fr_bares)) in enumerate(sorted_data.items()):
        fr_geoms_ghz = np.array(fr_geoms) * 1e-9
        fr_bares_ghz = np.array(fr_bares) * 1e-9
        label = f"{round(thickness * 1e9)} nm"
        ax.scatter(fr_geoms_ghz, fr_bares_ghz, label=label, color=blues[idx])

    fr_dummy_ghz = np.linspace(0, 30, 100)
    ax.plot(fr_dummy_ghz, fr_dummy_ghz, ls="--", c="k")

    ax.set_xlabel(r"Simulated $f_{\mathrm{r}}$ (GHz)")
    ax.set_ylabel(r"Measured $f_{\mathrm{r}}$ (GHz)")

    ax.set_xlim(3.5, 31)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.set_ylim(3.5, 8.25)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    plt.tight_layout()

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")

    plt.show()
