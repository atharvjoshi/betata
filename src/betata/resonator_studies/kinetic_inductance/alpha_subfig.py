"""
Generate subfigure composed of two plots with a common legend:
1) fr_geom vs fr
2) alpha vs pitch
with film thickness as the color series
"""

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

import numpy as np

from betata import plt, get_blues
from betata.resonator_studies.resonator import Resonator, load_resonators


if __name__ == "__main__":
    """ """

    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    figsavepath = resonator_folder / "alpha_pitch_fr_thickness.png"

    resonators: list[Resonator] = load_resonators()
    data_freq = defaultdict(dict)  # key=thickness, {key=fr_bare, value=fr_geom}
    data_alpha = defaultdict(dict)

    # only use CPW resonators from certain films for this subfigure, for neatness
    included_films = ["F1", "F2", "F5", "F8", "F9", "F11", "F14"]
    for resonator in resonators:
        film_name = resonator.name.split("_")[1]
        if resonator.type == "CPW" and film_name in included_films:
            thickness = resonator.film_thickness

            data_freq[thickness][resonator.fr_bare] = resonator.fr_geom
            data_alpha[thickness][resonator.pitch] = resonator.alpha_bare

    # create subfigure
    fig, (ax_freq, ax_alpha) = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    num_series = len(data_freq.keys())
    blues = get_blues(num_series, start=0.35, stop=0.95)

    # make fr_geom vs fr_bare subplot
    sorted_data_freq = dict(sorted(data_freq.items()))
    for idx, (thickness, inner_dict) in enumerate(sorted_data_freq.items()):
        fr_bares_ghz = np.array(list(inner_dict.keys())) * 1e-9
        fr_geoms_ghz = np.array(list(inner_dict.values())) * 1e-9
        label = f"{thickness * 1e6:.2f}"
        ax_freq.scatter(fr_geoms_ghz, fr_bares_ghz, label=label, color=blues[idx])

    fr_dummy_ghz = np.linspace(0, 30, 100)
    ax_freq.plot(fr_dummy_ghz, fr_dummy_ghz, ls="--", c="k")

    ax_freq.set_xlabel(r"$f_{\mathrm{r, geom}}$ (GHz)")
    ax_freq.set_ylabel(r"$f_{\mathrm{r}}$ (GHz)")

    ax_freq.set_xlim(3, 30)
    ax_freq.set_ylim(3, 8)

    ax_freq.set_xticks([3, 5, 10, 15, 20, 25, 30])
    ax_freq.set_yticks([3, 4, 5, 6, 7, 8])

    # make alpha vs pitch vs thickness subplot
    sorted_data_alpha = dict(sorted(data_alpha.items()))
    for idx, (thickness, inner_dict) in enumerate(sorted_data_alpha.items()):
        pitches_um = np.array(list(inner_dict.keys())) * 1e6
        alphas = np.array(list(inner_dict.values()))
        ax_alpha.scatter(pitches_um, alphas, color=blues[idx])

    ax_alpha.set_xlabel("CPW gap (μm)")
    ax_alpha.set_ylabel(r"KI fraction $\alpha$")

    ax_alpha.set_xlim(0, 17)
    ax_alpha.set_ylim(0, 1)

    ax_alpha.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    ax_alpha.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # add common legend
    handles, labels = ax_freq.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncols=num_series,
        loc="upper center",
        bbox_to_anchor=(0.0, 1.14, 1.0, 0.102),
        mode="expand",
        title="Film thickness (μm)",
        borderaxespad=0.5,
        frameon=False,
    )

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")
    plt.show()
