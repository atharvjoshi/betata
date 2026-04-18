""" """

from pathlib import Path

import numpy as np
from matplotlib import ticker

from betata import plt
from betata.qubit_measurements.qubit import load_qubit

T1_TRACE_COLOR = "#E77500"
T2R_TRACE_COLOR = "#2AA198"
T2E_TRACE_COLOR = "#003D7C"
TRANSPARENCY = 0.85

QUBITS_TO_INCLUDE = [
    "Q3_2p73",
    # "Q4_2p90",
    # "Q7_4p83",
    # "Q6_4p69",
]
NUM_SAMPLES = 333
YLIM_ADJUST = [50, 50]
TEXT_SIZE = 16

if __name__ == "__main__":
    """ """

    qubit_folder = Path(__file__).parents[3] / "out/qubit_measurements"

    for idx, qubit_name in enumerate(QUBITS_TO_INCLUDE):
        qubit_file = qubit_folder / f"{qubit_name}.h5"
        qubit = load_qubit(qubit_file)

        figsavepath = qubit_folder / f"{qubit_name}_T1_T2R_T2E_vs_time.svg"

        ej_ec_ratio = int(np.round(qubit.Ej / qubit.Ec))

        t1_us = qubit.t1 * 1e6
        t1_err_us = qubit.t1_err * 1e6
        t1_avg_us, t1_avg_err_us = np.mean(t1_us), np.std(t1_us)
        t1_max_idx = [np.argmax(t1_us)]

        t1_timestamp_day = qubit.t1_timestamp / 86400

        t2r_us = qubit.t2r * 1e6
        t2r_err_us = qubit.t2r_err * 1e6
        t2r_avg_us, t2r_avg_err_us = np.mean(t2r_us), np.std(t2r_us)
        t2r_max_idx = [np.argmax(t2r_us)]

        t2r_timestamp_day = qubit.t2r_timestamp / 86400

        t2e_us = qubit.t2e * 1e6
        t2e_err_us = qubit.t2e_err * 1e6
        t2e_avg_us, t2e_avg_err_us = np.mean(t2e_us), np.std(t2e_us)
        t2e_max_idx = [np.argmax(t2e_us)]

        t2e_timestamp_day = qubit.t2e_timestamp / 86400

        rng = np.random.default_rng(seed=4)
        t1_idxs = rng.choice(len(t1_timestamp_day), size=NUM_SAMPLES, replace=False)
        t2r_idxs = rng.choice(len(t2r_timestamp_day), size=NUM_SAMPLES, replace=False)
        t2e_idxs = rng.choice(len(t2e_timestamp_day), size=NUM_SAMPLES, replace=False)

        t1_sel_idxs = np.concatenate((t1_idxs, t1_max_idx))
        t1_sel_idxs.sort()
        t2r_sel_idxs = np.concatenate((t2r_idxs, t2r_max_idx))
        t2r_sel_idxs.sort()
        t2e_sel_idxs = np.concatenate((t2e_idxs, t2e_max_idx))
        t2e_sel_idxs.sort()

        t1_timestamp_day = t1_timestamp_day[t1_sel_idxs]
        t2r_timestamp_day = t2r_timestamp_day[t2r_sel_idxs]
        t2e_timestamp_day = t2e_timestamp_day[t2e_sel_idxs]

        t1_us = t1_us[t1_sel_idxs]
        t1_err_us = t1_err_us[t1_sel_idxs]
        t2r_us = t2r_us[t2r_sel_idxs]
        t2r_err_us = t2r_err_us[t2r_sel_idxs]
        t2e_us = t2e_us[t2e_sel_idxs]
        t2e_err_us = t2e_err_us[t2e_sel_idxs]

        fig, ax = plt.subplots(figsize=(10, 5))

        # ax.text(
        #    0.90,
        #    1.0,
        #    r"$E_\mathrm{J} \, / \, E_\mathrm{C} = $" + f"{ej_ec_ratio}",
        #    horizontalalignment="center",
        #    verticalalignment="center",
        #    transform=ax.transAxes,
        #    fontsize=TEXT_SIZE,
        # )

        # plot T on y axis
        ax.errorbar(
            t1_timestamp_day,
            t1_us,
            yerr=t1_err_us,
            color=T1_TRACE_COLOR,
            marker="o",
            ls="",
            zorder=-1,
            alpha=TRANSPARENCY,
        )
        ax.axhline(t1_avg_us, ls="--", color=T1_TRACE_COLOR)
        t1_avg_str = (
            r"$\mathrm{\overline{T}_1}$"
            + f" = {t1_avg_us:.0f} ± {t1_avg_err_us:.0f} μs"
        )
        ax.text(
            0.5, #0.025,
            1.0, #1.0,
            t1_avg_str,
            horizontalalignment="center", #"left",
            verticalalignment="center",
            transform=ax.transAxes,
            color=T1_TRACE_COLOR,
            # fontsize=TEXT_SIZE,
        )

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
            r"$\mathrm{\overline{T}_{2, R}}$"
            + f" = {t2r_avg_us:.0f} ± {t2r_avg_err_us:.0f} μs"
        )
        ax.text(
            0.5, #0.5,
            0.8, #1.0,
            t2r_avg_str,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color=T2R_TRACE_COLOR,
            # fontsize=TEXT_SIZE,
        )

        ax.errorbar(
            t2e_timestamp_day,
            t2e_us,
            yerr=t2e_err_us,
            color=T2E_TRACE_COLOR,
            marker="o",
            ls="",
            zorder=-1,
            alpha=TRANSPARENCY,
        )
        ax.axhline(t2e_avg_us, ls="--", color=T2E_TRACE_COLOR)
        t2e_avg_str = (
            r"$\mathrm{\overline{T}_{2, E}}$"
            + f" = {t2e_avg_us:.0f} ± {t2e_avg_err_us:.0f} μs"
        )
        ax.text(
            0.5, #1.0,
            0.9, #1.0,
            t2e_avg_str,
            horizontalalignment="center", #"right",
            verticalalignment="center",
            transform=ax.transAxes,
            color=T2E_TRACE_COLOR,
            # fontsize=TEXT_SIZE,
        )

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
        y_min = min(min(t1_us), min(t2r_us), min(t2e_us)) - YLIM_ADJUST[0]
        y_max = max(max(t1_us), max(t2r_us), max(t2e_us)) + YLIM_ADJUST[1]
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Time (day)")
        ax.set_ylabel(
            r"$\{ \mathrm{T_1}$ , $\mathrm{T_{2, R}}$ , $\mathrm{T_{2, E}} \}$ (μs)"
        )

        plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

        plt.show()
