"""

Run the script "alpha_bare.py" to find and save `alpha` before running this script.

"""

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from betata import plt, get_blues
from betata.resonator_studies.resonator import Resonator, load_resonators

if __name__ == "__main__":
    """ """

    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    figsavepath = resonator_folder / "alpha_vs_pitch.png"

    resonators: list[Resonator] = load_resonators()
    data = defaultdict(dict)

    # only use CPW resonators from certain films for this subfigure, for neatness
    included_films = ["F1", "F2", "F5", "F8", "F9", "F11", "F14"]
    for resonator in resonators:
        film_name = resonator.name.split("_")[1]
        if resonator.type == "CPW" and film_name in included_films:
            thickness = resonator.film_thickness
            pitch = resonator.pitch
            alpha = resonator.alpha_bare
            alpha_err = resonator.alpha_bare_err
            data[thickness][pitch] = (alpha, alpha_err)

    sorted_data = dict(sorted(data.items()))

    fig, ax = plt.subplots(figsize=(10, 6))
    blues = get_blues(len(sorted_data.keys()))

    for idx, (thickness, inner_dict) in enumerate(sorted_data.items()):
        pitches, alphas, alpha_errs = [], [], []

        for pitch, (alpha, alpha_err) in inner_dict.items():
            pitches.append(pitch)
            alphas.append(alpha)
            alpha_errs.append(alpha_err)

        pitches_um = np.array(pitches) * 1e6
        alphas, alpha_errs = np.array(alphas), np.array(alpha_errs)
        label = f"{round(thickness * 1e9)}nm"
    
        #ax.errorbar(
        #    pitches_um,
        #    alphas,
        #    yerr=alpha_errs,
        #    ls="",
        #    marker="o",
        #    label=label,
        #    c=blues[idx],
        #)

        ax.scatter(pitches_um, alphas, label=label, c=blues[idx])

    ax.set(xlabel=r"CPW gap ($\mu$m)", ylabel=r"KI fraction $\alpha$")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(bbox_to_anchor=(1.05, 1))

    fig.tight_layout()
    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")
    plt.show()
