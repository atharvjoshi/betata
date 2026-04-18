"""
Select 3 exemplary circle fits showing high Q_int and high L_k/sq to showcase in the SI
"""

from pathlib import Path

import numpy as np
from rrfit.fitfns import rr_s21_hanger
from matplotlib import ticker

from betata import plt, get_purples
from betata.resonator_studies.trace import Trace, load_traces, load_fitted_traces

BTA_COLOR = get_purples(1, 1.0, 1.0)[0]
TRANSPARENCY = 0.85

if __name__ == "__main__":
    """ """

    resonator_names = ["R2_F1_4p47", "R13_F2_5p72", "R78_F12_3p85"]
    outer_folder = Path(__file__).parents[4]
    resonator_folder = outer_folder / "out/resonator_studies"

    selected_powers = [-90, -90, -80]  # dBm
    selected_temps = [(8e-3, 12e-3), (12e-3, 18e-3), (8e-3, 12e-3)]

    tick_spacings = [0.4, 0.2, 0.2]

    for idx, resonator_name in enumerate(resonator_names):
        data_folder = outer_folder / f"data/resonator_studies/{resonator_name}"
        resonator_file = resonator_folder / f"{resonator_name}.h5"
        figsavepath = resonator_folder / f"{resonator_name}_circle_fit.svg"

        raw_traces: list[Trace] = load_traces(data_folder)
        fitted_traces: list[Trace] = load_fitted_traces(resonator_file)

        raw_trace, fitted_trace = None, None
        for trace in raw_traces:
            min_t, max_t = selected_temps[idx][0], selected_temps[idx][1]
            selected_power = selected_powers[idx]
            if trace.power == selected_power and min_t < trace.temperature < max_t:
                raw_trace = trace
        for trace in fitted_traces:
            if trace.filename == raw_trace.filename:
                fitted_trace = trace

        print(fitted_trace.Qi, fitted_trace.Qi_err)
        print(fitted_trace.temperature)

        frequency = raw_trace.frequency
        s21_raw = raw_trace.s21real + 1j * raw_trace.s21imag
        s21_nodelay = s21_raw * np.exp(-1j * 2 * np.pi * frequency * raw_trace.tau)
        orp = fitted_trace.background_amp * np.exp(1j * fitted_trace.background_phase)
        s21_canonical = s21_nodelay / orp

        s21_to_plot = s21_canonical[::2]  # downsample for clarity

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

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(
            s21_to_plot.real,
            s21_to_plot.imag,
            color=BTA_COLOR,
            label="data",
            alpha=TRANSPARENCY,
        )
        ax.plot(s21_fit.real, s21_fit.imag, c=BTA_COLOR, label="model")

        ax.set_xlabel(r"Re($\mathrm{S_{21}}$)")
        ax.set_ylabel(r"Im($\mathrm{S_{21}}$)")

        tick_spacing = tick_spacings[idx]
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.set_aspect("equal", "datalim")

        fig.tight_layout()

        plt.savefig(figsavepath, dpi=600, bbox_inches="tight")
        plt.show()
