"""subfigure"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from uncertainties import ufloat

from betata import plt, get_blues, get_purples
from betata.resonator_studies.resonator import load_resonators

ATA_COLOR = "darkorange" #"#FF7900"
BTA_COLOR = get_purples(1, 1.0, 1.0)[0]
SAPPHIRE_COLOR = "k" #"#474A51"
FILL_TRANSPARENCY = 0.25

ATA_TAN_DELTA = ufloat(8.1e-4, 0.6e-4)
BTA_TAN_DELTA = ufloat(1.3e-3, 0.1e-3)  # all thicknesses
SAPPHIRE_TAN_DELTA = ufloat(1.3e-7, 0.2e-7)
P_SUB = 0.90  # assume negligible variation across resonator types/pitches

REJECTION_THRESHOLD = 0.75  # relative error

LABEL_LIST = ["55 nm", "100-130 nm", "200-240 nm", "400-480 nm", "1000 nm"]
COLOR_LIST = [
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
]
BLUES = get_blues(values=[0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])


def fit_delta_surf_sub(x, tan_delta_surf, tan_delta_sub, p_sub):
    """ """
    return 1 / (x * tan_delta_surf + p_sub * tan_delta_sub)


def plot_data(
    data,
    figsize=(10, 7),
    ylim=(1e5, 2e7),
    xlim=(3e-3, 0.5e-4),
):
    """ """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"Surface participation ratio ($p_{\mathrm{MS}})$")
    ax.set_ylabel(r"$Q_{\mathrm{TLS,0}}$")

    for thickness, lists in data.items():
        if 0.0 < thickness < 0.025:
            color = BLUES[0]
        elif 0.025 < thickness < 0.06:
            color = BLUES[1]
        elif 0.06 < thickness < 0.15:
            color = BLUES[2]
        elif 0.15 < thickness < 0.3:
            color = BLUES[3]
        elif 0.3 < thickness < 0.5:
            color = BLUES[4]
        elif 0.5 < thickness < 1.2:
            color = BLUES[5]
        else:
            color = BLUES[6]

        ax.errorbar(
            lists["p_ms"],
            lists["q_tls0"],
            yerr=lists["q_tls0_err"],
            c=color,
            ls="",
            marker="o",
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # prevent duplicate labels
    #handles, labels = plt.gca().get_legend_handles_labels()
    #unique_labels = dict(zip(labels, handles))
    #leg = plt.legend(unique_labels.values(), unique_labels.keys(), loc="lower right")
    #ax.add_artist(leg)

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
    (line,) = axis.plot(p_ms, q_tls0, **plot_params)
    if tan_delta_err is not None:
        axis.fill_between(
            p_ms,
            1 / (p_ms * (tan_delta - tan_delta_err / 2)),
            1 / (p_ms * (tan_delta + tan_delta_err / 2)),
            color=plot_params["color"],
            alpha=FILL_TRANSPARENCY,
        )
    return line


def add_bulk_loss_tangent(
    tan_delta,
    axis,
    tan_delta_err=None,
    xlim=None,
    **plot_params,
):
    """ """
    line = axis.axhline(1 / tan_delta, **plot_params)
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
    return line


if __name__ == "__main__":
    """ """

    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    resonators = load_resonators()
    figsavepath = resonator_folder / "wang_plot_by_thickness.svg"

    min_thickness, max_thickness = -np.inf, np.inf

    # key: thickness, value: dict of lists p_ms, q_tls0, q_tls0_err
    data = defaultdict(lambda: defaultdict(list))
    for resonator in resonators:
        if resonator.qpt_fit_params is None:
            continue

        thickness = resonator.film_thickness * 1e6
        if not min_thickness < thickness < max_thickness:
            continue

        q_tls0_param = resonator.qpt_fit_params["Q_TLS0"]

        if q_tls0_param["stderr"] / q_tls0_param["value"] > REJECTION_THRESHOLD:
            continue

        data[thickness]["q_tls0"].append(q_tls0_param["value"])
        data[thickness]["q_tls0_err"].append(q_tls0_param["stderr"])
        data[thickness]["p_ms"].append(resonator.p_ms)

    p_ms_lim = (3e-3, 0.5e-4)
    q_tls0_lim = (1e5, 1.5e7)

    figure, axis = plot_data(
        data,
        xlim=p_ms_lim,
        ylim=q_tls0_lim,
    )

    line1 = add_bulk_loss_tangent(
        SAPPHIRE_TAN_DELTA.n,
        axis,
        color=SAPPHIRE_COLOR,
        ls="--",
        label="Bulk",
        tan_delta_err=SAPPHIRE_TAN_DELTA.s,
        xlim=p_ms_lim,
    )

    line2 = add_surface_loss_tangent(
        ATA_TAN_DELTA.n,
        axis,
        *p_ms_lim,
        tan_delta_err=ATA_TAN_DELTA.s,
        color=ATA_COLOR,
        ls="--",
        label=r"$\alpha\mathrm{-Ta}$",
    )

    line3 = add_surface_loss_tangent(
        BTA_TAN_DELTA.n,
        axis,
        *p_ms_lim,
        #tan_delta_err=BTA_TAN_DELTA.s,
        color=BTA_COLOR,
        #ls="--",
        label=r"$\beta\mathrm{-Ta}$",
    )

    axis.legend(handles=[line1, line2, line3], frameon=False)
    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
