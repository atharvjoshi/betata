"""subfigure"""

from pathlib import Path

import lmfit
import numpy as np
from uncertainties import ufloat

from betata import plt
from betata.resonator_studies.resonator import Resonator, load_resonator

ATA_COLOR = "#FF7900"
BTA_COLOR = "#762A83"
SAPPHIRE_COLOR = "#474A51"
FILL_TRANSPARENCY = 0.25

ATA_TAN_DELTA = ufloat(8.1e-4, 0.6e-4)
SAPPHIRE_TAN_DELTA = ufloat(1.3e-7, 0.2e-7)
P_SUB = 0.90  # assume negligible variation across resonator types/pitches

REJECTION_THRESHOLD = 0.75  # relative error


def load_resonators(folder: Path) -> list[Resonator]:
    """ """
    resonators = []
    for file in folder.iterdir():
        if file.suffix not in [".h5", ".hdf5"]:
            continue

        resonator = load_resonator(file)
        resonators.append(resonator)
    return resonators


def fit_delta_surf_sub(x, tan_delta_surf, tan_delta_sub, p_sub):
    """ """
    return 1 / (x * tan_delta_surf + p_sub * tan_delta_sub)


def plot_data(x, y, yerr, figsize=(10, 7), ylim=(1e5, 2e7), xlim=(3e-3, 0.5e-4)):
    """ """

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()

    ax.set_xlabel(r"Surface participation ratio ($p_{\mathrm{MS}})$")
    ax.set_ylabel(r"$Q_{\mathrm{TLS,0}}$")

    ax.errorbar(x, y, yerr=yerr, ls="", c=BTA_COLOR, marker="o")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    fig.tight_layout()

    return fig, ax


def add_surface_loss_tangent(
    tan_delta,
    axis,
    xmin,
    xmax,
    tan_delta_err=None,
    **plot_params,
):
    """ """
    p_ms = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    q_tls0 = 1 / (p_ms * tan_delta)
    axis.plot(p_ms, q_tls0, **plot_params)
    if tan_delta_err is not None:
        axis.fill_between(
            p_ms,
            1 / (p_ms * (tan_delta - tan_delta_err / 2)),
            1 / (p_ms * (tan_delta + tan_delta_err / 2)),
            color=plot_params["color"],
            alpha=FILL_TRANSPARENCY,
        )
    return axis


def add_bulk_loss_tangent(
    tan_delta,
    axis,
    tan_delta_err=None,
    xlim=None,
    **plot_params,
):
    """ """
    axis.axhline(1 / tan_delta, **plot_params)
    if tan_delta_err is not None and xlim is not None:
        xmin, xmax = xlim
        p_ms = np.logspace(np.log10(xmin), np.log10(xmax), 100)
        axis.fill_between(
            p_ms,
            1 / (tan_delta - tan_delta_err / 2),
            1 / (tan_delta + tan_delta_err / 2),
            color=plot_params["color"],
            alpha=FILL_TRANSPARENCY,
        )
    return axis


if __name__ == "__main__":
    """ """

    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    resonators = load_resonators(resonator_folder)
    figsavepath = resonator_folder / "wang_plot_100-500nm.png"

    min_thickness, max_thickness = 90e-9, 500e-9

    p_ms, q_tls0, q_tls0_err = [], [], []
    for resonator in resonators:
        if resonator.qpt_fit_params is None:
            continue

        if not min_thickness < resonator.film_thickness < max_thickness:
            continue

        q_tls0_param = resonator.qpt_fit_params["Q_TLS0"]

        if q_tls0_param["stderr"] / q_tls0_param["value"] > REJECTION_THRESHOLD:
            continue

        q_tls0.append(q_tls0_param["value"])
        q_tls0_err.append(q_tls0_param["stderr"])
        p_ms.append(resonator.p_ms)
    p_ms, q_tls0, q_tls0_err = np.array(p_ms), np.array(q_tls0), np.array(q_tls0_err)

    p_ms_lim = (3e-3, 0.5e-4)
    q_tls0_lim = (1e5, 2e7)
    figure, axis = plot_data(
        p_ms,
        q_tls0,
        q_tls0_err,
        xlim=p_ms_lim,
        ylim=q_tls0_lim,
    )

    model = lmfit.Model(fit_delta_surf_sub)
    params = model.make_params(
        tan_delta_surf={"value": 1e-3},
        tan_delta_sub={"value": SAPPHIRE_TAN_DELTA.n, "vary": False},
        p_sub={"value": P_SUB, "vary": False},
    )
    result = model.fit(
        q_tls0,
        x=p_ms,
        params=params,
        weights=1 / q_tls0_err,
        method="least_squares",
    )

    print(result.fit_report())
    print(result.params.pretty_print())

    axis = add_bulk_loss_tangent(
        SAPPHIRE_TAN_DELTA.n,
        axis,
        color=SAPPHIRE_COLOR,
        ls="--",
        label="Bulk",
        tan_delta_err=SAPPHIRE_TAN_DELTA.s,
        xlim=p_ms_lim,
    )

    axis = add_surface_loss_tangent(
        ATA_TAN_DELTA.n,
        axis,
        *p_ms_lim,
        tan_delta_err=ATA_TAN_DELTA.s,
        color=ATA_COLOR,
        ls="--",
        label=r"$\alpha\mathrm{-Ta}$",
    )

    bta_tan_delta = result.params["tan_delta_surf"].value
    bta_tan_delta_err = result.params["tan_delta_surf"].stderr
    axis = add_surface_loss_tangent(
        bta_tan_delta,
        axis,
        *p_ms_lim,
        tan_delta_err=bta_tan_delta_err,
        color=BTA_COLOR,
        ls="--",
        label=r"$\beta\mathrm{-Ta}$",
    )

    plt.legend(loc="lower right")
    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")
    plt.show()
