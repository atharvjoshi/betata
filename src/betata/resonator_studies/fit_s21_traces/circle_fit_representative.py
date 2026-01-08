"""
Generate a representative circle fit subfigure
"""

from pathlib import Path

import numpy as np
from rrfit.fitfns import rr_s21_hanger

from betata import plt, get_purples
from betata.resonator_studies.trace import Trace, load_traces, load_fitted_traces

BTA_COLOR = get_purples(1, 1.0, 1.0)[0]
TRANSPARENCY = 0.85

if __name__ == "__main__":
    """ """

    resonator_name = "R70_F11_5p59"

    data_folder = Path(__file__).parents[4] / f"data/resonator_studies/{resonator_name}"
    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    resonator_file = resonator_folder / f"{resonator_name}.h5"

    figsavepath = resonator_folder / "circle_fit.png"

    raw_traces: list[Trace] = load_traces(data_folder)
    fitted_traces: list[Trace] = load_fitted_traces(resonator_file)

    selected_power, selected_temp = -90, 15e-3
    raw_trace, fitted_trace = None, None
    for trace in raw_traces:
        if trace.power == selected_power and trace.temperature < selected_temp:
            raw_trace = trace
    for trace in fitted_traces:
        if trace.filename == raw_trace.filename:
            fitted_trace = trace

    frequency = raw_trace.frequency
    s21_raw = raw_trace.s21real + 1j * raw_trace.s21imag
    s21_nodelay = s21_raw * np.exp(-1j * 2 * np.pi * frequency * raw_trace.tau)
    orp = fitted_trace.background_amp * np.exp(1j * fitted_trace.background_phase)
    s21_canonical = s21_nodelay / orp

    s21_to_plot = s21_canonical[::3]  # downsample for clarity

    s21_fit = rr_s21_hanger(
        frequency,
        fitted_trace.fr,
        fitted_trace.Ql,
        fitted_trace.absQc,
        fitted_trace.phi,
        a=1,
        alpha=0,
        tau=0,
    )

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.scatter(
        s21_to_plot.real,
        s21_to_plot.imag,
        color=BTA_COLOR,
        label="data",
        alpha=TRANSPARENCY,
    )
    ax.plot(s21_fit.real, s21_fit.imag, c=BTA_COLOR, label="model")

    ax.set_xlabel(r"Re($\mathrm{S_{21}}$)", fontsize=24)
    ax.set_ylabel(r"Im($\mathrm{S_{21}}$)", fontsize=24)

    ax.set_aspect("equal", "datalim")

    ax.set_xticks([0.5, 1.0])
    ax.set_yticks([0, -0.5])

    # make inset
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", width=1.5, labelsize=24)

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")
    plt.show()
