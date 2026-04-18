"""

Run the script "alpha_bare_fr_lkin.py" to find and save `alpha` before running this script.

"""

from collections import defaultdict
from pathlib import Path

import lmfit
import numpy as np
from matplotlib import ticker
from scipy.constants import mu_0
from scipy.optimize import fsolve
from scipy.special import ellipk

from betata import plt, get_blues
from betata.resonator_studies.resonator import Resonator, load_resonators

CHAR_IMP = 50
EPSILON_EFF = (1 + 10.4) / 2  # sapphire
SUBSTRATE_HEIGHT = 530e-6

BLUES = get_blues(values=[0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])
TRANSPARENCY = 1.0


def l_geom_model(w, s):
    """w: CPW width, s: CPW pitch"""
    k = w / (w + 2 * s)
    kprime = np.sqrt(1 - k**2)
    l_geom = (mu_0 / 4) * (ellipk(kprime) / ellipk(k))
    return l_geom


def alpha_model(A, pitch, width, l_geom):
    """A: dimensionless geometric factor"""
    l_kin = A / (width + 2 * pitch)
    return l_kin / (l_kin + l_geom)


def alpha_error_fn(params, data, pitch, width, l_geom):
    """ """
    geometric_factor = params["A"].value
    return data - alpha_model(geometric_factor, pitch, width, l_geom)


def impedance(w, s, h=SUBSTRATE_HEIGHT):
    """ """
    b = w + 2 * s
    k = w / b
    kprime = np.sqrt(1 - k**2)
    k3 = np.tanh((np.pi * w) / (4 * h)) / np.tanh((np.pi * b) / (4 * h))
    k3prime = np.sqrt(1 - k3**2)

    epsilon_eff = EPSILON_EFF
    prefactor = (60 * np.pi) / np.sqrt(epsilon_eff)

    return prefactor / ((ellipk(k) / ellipk(kprime)) + (ellipk(k3) / ellipk(k3prime)))


def target_width_model(width, pitch):
    """ """
    return CHAR_IMP - impedance(width, pitch)


if __name__ == "__main__":
    """ """

    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    figsavepath = resonator_folder / "alpha_vs_pitch_vs_thickness.svg"

    resonators: list[Resonator] = load_resonators()
    data = defaultdict(list)

    alpha_data = defaultdict(dict)
    N_sq_data = defaultdict(dict)

    # only use CPW resonators from certain films for this subfigure, for neatness
    included_films = ["F1", "F2", "F5", "F8", "F9", "F11", "F14"]
    for resonator in resonators:
        film_name = resonator.name.split("_")[1]
        if resonator.type == "CPW" and film_name in included_films:
            thickness = resonator.film_thickness
            data[thickness].append(resonator)

    fig, ax = plt.subplots(figsize=(7, 5))

    for resonators in data.values():
        resonators.sort(key=lambda x: x.pitch)

    sorted_data = dict(sorted(data.items()))

    for idx, (thickness, resonators) in enumerate(sorted_data.items()):
        pitches = np.array([resonator.pitch for resonator in resonators])
        widths = np.array([resonator.width for resonator in resonators])
        l_geoms = np.array([2 * rr.l_geom / rr.length for rr in resonators])
        channel_widths = widths + 2 * pitches
        alphas = np.array([resonator.alpha_bare for resonator in resonators])

        pitches_um = pitches * 1e6
        label = f"{round(thickness * 1e9)}nm"

        fit_params = lmfit.Parameters()
        fit_params.add("A", value=1e-12)
        fit_result = lmfit.minimize(
            alpha_error_fn,
            fit_params,
            args=(alphas, pitches, widths, l_geoms),
        )
        print(lmfit.fit_report(fit_result))
        geometric_factor = fit_result.params["A"].value

        pitches_dummy = np.linspace(min(pitches), max(pitches), 101)
        widths_dummy = []
        for pitch in pitches_dummy:
            target_width = fsolve(target_width_model, pitch * 2, args=(pitch,))[0]
            widths_dummy.append(target_width)
        widths_dummy = np.array(widths_dummy)
        l_geoms_dummy = l_geom_model(widths_dummy, pitches_dummy)
        alphas_dummy = alpha_model(
            geometric_factor,
            pitches_dummy,
            widths_dummy,
            l_geoms_dummy,
        )
        pitches_dummy_um = pitches_dummy * 1e6

        # ax.errorbar(
        #    pitches_um,
        #    alphas,
        #    yerr=alpha_errs,
        #    ls="",
        #    marker="o",
        #    label=label,
        #    c=blues[idx],
        # )

        ax.scatter(
            pitches_um,
            alphas,
            label=label,
            color=BLUES[idx],
            alpha=TRANSPARENCY,
        )
        ax.plot(pitches_dummy_um, alphas_dummy, color=BLUES[idx])

    ax.set(xlabel=r"CPW gap width, $s$ (μm)", ylabel=r"$\alpha$")

    ax.set_xlim(1, 17)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    #ax.legend(bbox_to_anchor=(1.05, 1))

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
