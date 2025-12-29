"""
Plot simulated frequency (with no sheet inductance) `f_geom` against measured bare frequency 'fr_bare'.

Run the script "alpha_bare.py" to find and save `fr_bare` before running this script.

"""

from pathlib import Path

from brokenaxes import brokenaxes
import numpy as np

from betata import plt, get_blues
from betata.resonator_studies.resonator import Resonator, load_resonators


if __name__ == "__main__":
    """ """

    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    figsavepath = resonator_folder / "fr_geom_vs_fr.png"

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

    fig = plt.figure(figsize=(10, 6))
    bax: plt.Axes = brokenaxes(
        xlims=((0, 0.01), (2, 30)),
        ylims=((0, 0.01), (2, 10)),
    )
    blues = get_blues(len(sorted_data.keys()))

    for idx, (thickness, (fr_geoms, fr_bares)) in enumerate(sorted_data.items()):
        fr_geoms_ghz = np.array(fr_geoms) * 1e-9
        fr_bares_ghz = np.array(fr_bares) * 1e-9
        label = f"{round(thickness * 1e9)} nm"
        bax.scatter(fr_geoms_ghz, fr_bares_ghz, label=label, c=blues[idx])

    fr_dummy_ghz = np.linspace(0, 30, 100)
    bax.plot(fr_dummy_ghz, fr_dummy_ghz, ls="--", c="k")

    bax.set_xlabel(r"$f_{\mathrm{r, geom}}$ (GHz)", labelpad=40)
    bax.set_ylabel(r"$f_{\mathrm{r}}$ (GHz)")

    bax.legend(bbox_to_anchor=(1.55, 1))

    # do this to make tight_layout work with brokenaxes
    plt.tight_layout()
    for handle in bax.diag_handles:
        handle.remove()
    bax.draw_diags()

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")
    plt.show()
