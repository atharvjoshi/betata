""" """

from pathlib import Path

import numpy as np
from matplotlib import ticker

from betata import plt
from betata.qubit_measurements.qubit import load_qubit

T2R_TRACE_COLOR = "#2AA198"
TRANSPARENCY = 0.85

QUBITS_TO_INCLUDE = [
    "Q3_2p73",
    # "Q6_4p69",
]
NUM_SAMPLES = 500

if __name__ == "__main__":
    """ """

    qubit_folder = Path(__file__).parents[3] / "out/qubit_measurements"

    for idx, qubit_name in enumerate(QUBITS_TO_INCLUDE):
        qubit_file = qubit_folder / f"{qubit_name}.h5"
        qubit = load_qubit(qubit_file)

        figsavepath = qubit_folder / f"{qubit_name}_T2R_vs_time.svg"

        t2r_us = qubit.t2r * 1e6
        t2r_err_us = qubit.t2r_err * 1e6
        t2r_avg_us, t2r_avg_err_us = np.mean(t2r_us), np.std(t2r_us)
        t2r_timestamp_day = qubit.t2r_timestamp / 86400

        # downsample data for figure clarity
        t2r_max_idx = [np.argmax(t2r_us)]

        rng = np.random.default_rng(seed=4)
        sel_idxs = rng.choice(len(t2r_timestamp_day), size=NUM_SAMPLES, replace=False)

        t2r_sel_idxs = np.concatenate((sel_idxs, t2r_max_idx))
        t2r_sel_idxs.sort()

        t2r_timestamp_day = t2r_timestamp_day[t2r_sel_idxs]
        t2r_us = t2r_us[t2r_sel_idxs]
        t2r_err_us = t2r_err_us[t2r_sel_idxs]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.errorbar(
            t2r_timestamp_day,
            t2r_us,
            yerr=t2r_err_us,
            color=T2R_TRACE_COLOR,
            marker="o",
            ls="",
            zorder=-1,
            alpha=TRANSPARENCY,
        )
        ax.axhline(t2r_avg_us, ls="--", color=T2R_TRACE_COLOR)
        t2r_avg_str = (
            r"$\mathrm{\overline{T}_\mathrm{2, R}}$"
            + f" = {t2r_avg_us:.0f} ± {t2r_avg_err_us:.0f} μs"
        )

        ax.text(
            0.05,
            0.90,
            t2r_avg_str,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
            # color=T2R_TRACE_COLOR,
        )

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

        ax.set_xlabel("Time (day)")
        ax.set_ylabel(r"$\mathrm{T_{2, R}}$ (μs)")

        plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

        plt.show()
