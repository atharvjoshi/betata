""" """

from pathlib import Path

import numpy as np
from matplotlib import ticker

from betata import plt
from betata.qubit_measurements.qubit import load_qubit

T1_TRACE_COLOR = "#E77500"
T2E_TRACE_COLOR = "#003D7C"
TRANSPARENCY = 0.85

if __name__ == "__main__":
    """ """

    qubit_name = "Q6_4p69"

    out_folder = Path(__file__).parents[3] / "out"
    qubit_file = out_folder / f"qubit_measurements/{qubit_name}.h5"
    qubit = load_qubit(qubit_file)

    figsavepath = (
        out_folder / f"qubit_measurements/{qubit_name}_T1_T2E_vs_time_fig4.png"
    )

    t1_us = qubit.t1 * 1e6
    t1_err_us = qubit.t1_err * 1e6
    t1_avg_us, t1_avg_err_us = np.mean(t1_us), np.std(t1_us)

    t2e_us = qubit.t2e * 1e6
    t2e_err_us = qubit.t2e_err * 1e6
    t2e_avg_us, t2e_avg_err_us = np.mean(t2e_us), np.std(t2e_us)

    omega_q = 2 * np.pi * qubit.f_q

    omega_t1 = omega_q * qubit.t1
    omega_t1_err = omega_q * qubit.t1_err
    t1_timestamp_day = qubit.t1_timestamp / 86400

    omega_t1_million, omega_t1_err_million = omega_t1 * 1e-6, omega_t1_err * 1e-6

    omega_t2e = omega_q * qubit.t2e
    omega_t2e_err = omega_q * qubit.t2e_err
    t2e_timestamp_day = qubit.t2e_timestamp / 86400

    omega_t2e_million, omega_t2e_err_million = omega_t2e * 1e-6, omega_t2e_err * 1e-6

    omega_t1_avg, omega_t1_avg_err = np.mean(omega_t1), np.std(omega_t1)
    omega_t2e_avg, omega_t2e_avg_err = np.mean(omega_t2e), np.std(omega_t2e)

    omega_t1_avg_million = omega_t1_avg * 1e-6
    omega_t2e_avg_million = omega_t2e_avg * 1e-6

    # downsample data for figure clarity
    omega_t1_max_idx = [np.argmax(omega_t1)]
    omega_t2e_max_idx = [np.argmax(omega_t2e)]

    # both t1 and t2e samples have same length 791
    # print(len(t1_timestamp_day))
    # print(len(t2e_timestamp_day))

    rng = np.random.default_rng(seed=4)
    sel_idxs = rng.choice(len(t1_timestamp_day), size=400, replace=False)

    t1_sel_idxs = np.concatenate((sel_idxs, omega_t1_max_idx))
    t1_sel_idxs.sort()
    t2e_sel_idxs = np.concatenate((sel_idxs, omega_t2e_max_idx))
    t2e_sel_idxs.sort()

    t1_timestamp_day = t1_timestamp_day[t1_sel_idxs]
    omega_t1_million = omega_t1_million[t1_sel_idxs]
    omega_t1_err_million = omega_t1_err_million[t1_sel_idxs]

    t2e_timestamp_day = t2e_timestamp_day[t2e_sel_idxs]
    omega_t2e_million = omega_t2e_million[t2e_sel_idxs]
    omega_t2e_err_million = omega_t2e_err_million[t2e_sel_idxs]

    t1_us = t1_us[t1_sel_idxs]
    t1_err_us = t1_err_us[t1_sel_idxs]
    t2e_us = t2e_us[t2e_sel_idxs]
    t2e_err_us = t2e_err_us[t2e_sel_idxs]

    fig, ax = plt.subplots(figsize=(10, 5))

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
        r"$\mathrm{\overline{T}_1}$" + f" = {t1_avg_us:.0f} ± {t1_avg_err_us:.0f} μs"
    )

    ax.text(
        0.35,
        0.8,
        t1_avg_str,
        horizontalalignment="right",
        verticalalignment="center",
        transform=ax.transAxes,
        color=T1_TRACE_COLOR,
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
        0.35,
        0.95,
        t2e_avg_str,
        horizontalalignment="right",
        verticalalignment="center",
        transform=ax.transAxes,
        color=T2E_TRACE_COLOR,
    )

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

    ax.set_xlabel("Time (day)")
    ax.set_ylabel(r"$\{ \mathrm{T_1}$ , $\mathrm{T_{2, E}} \}$ (μs)")

    """
    # plot omega * T on y axis

    ax.errorbar(
        t1_timestamp_day,
        omega_t1_million,
        yerr=omega_t1_err_million,
        color=T1_TRACE_COLOR,
        marker="o",
        ls="",
        zorder=-1,
    )
    ax.axhline(omega_t1_avg_million, ls="--", color=T1_TRACE_COLOR)

    ax.errorbar(
        t2e_timestamp_day,
        omega_t2e_million,
        yerr=omega_t2e_err_million,
        color=T2E_TRACE_COLOR,
        marker="o",
        ls="",
        zorder=-1,
    )
    ax.axhline(omega_t2e_avg_million, ls="--", color=T2E_TRACE_COLOR)

    ax.set_ylim(3, 17.5)
    ax.set_yticks([4, 8, 12, 16])

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))

    ax.set_xlabel("Time (day)")
    ax.set_ylabel(r"$\mathrm{\omega_q \times T}$ (million)")
    """

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
