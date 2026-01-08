""" """

from pathlib import Path

from brokenaxes import brokenaxes
import numpy as np
from matplotlib import ticker

from betata import plt, get_purples
from betata.qubit_measurements.qubit import load_qubit

TRACE_COLOR = get_purples(1, 0.9, 0.9)[0]
TRANSPARENCY = 0.85


if __name__ == "__main__":
    """ """

    qubit_name = "Q1_2p61"
    out_folder = Path(__file__).parents[4] / "out"
    qubit_file = out_folder / f"qubit_measurements/{qubit_name}.h5"

    figsavepath = out_folder / "qubit_measurements/T1_vs_time_max.png"

    qubit = load_qubit(qubit_file)
    omega_q = 2 * np.pi * qubit.f_q

    # return Q in million
    Q_functions = (lambda x: x * omega_q * 1e-12, lambda x: x / (omega_q * 1e-12))

    t1_us = qubit.t1 * 1e6
    t1_err_us = qubit.t1_err * 1e6
    t1_timestamp_day = qubit.t1_timestamp / 86400

    t1_avg_us = np.mean(t1_us)
    t1_avg_err_us = np.std(t1_us)
    t1_avg_str = (
        r"$\mathrm{\overline{T}_1}$" + f" = {t1_avg_us:.0f} ± {t1_avg_err_us:.0f} μs"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bax = brokenaxes(xlims=((-0.5, 2), (3, 7), (11, 12.5)), hspace=0.02)

    ax.set_axis_off()

    # downsample data for figure clarity
    t1_max_idx = [np.argmax(t1_us)]
    s1_all_idxs = np.argwhere(t1_timestamp_day < 2).flatten()
    s2_all_idxs = np.argwhere((t1_timestamp_day > 3) & (t1_timestamp_day < 7)).flatten()
    s3_all_idxs = np.argwhere(t1_timestamp_day > 11).flatten()

    s1_num_pts, s2_num_pts, s3_num_pts = 160, 240, 99

    s1_sel_idxs = np.random.choice(s1_all_idxs, size=s1_num_pts, replace=False)
    s2_sel_idxs = np.random.choice(s2_all_idxs, size=s1_num_pts, replace=False)
    s3_sel_idxs = np.random.choice(s3_all_idxs, size=s1_num_pts, replace=False)

    ds_idxs = np.concatenate((s1_sel_idxs, s2_sel_idxs, s3_sel_idxs, t1_max_idx))
    ds_idxs.sort()

    t1_timestamp_day_downsampled = t1_timestamp_day[ds_idxs]
    t1_us_downsampled = t1_us[ds_idxs]
    t1_err_us_downsampled = t1_err_us[ds_idxs]

    bax.errorbar(
        t1_timestamp_day_downsampled,
        t1_us_downsampled,
        yerr=t1_err_us_downsampled,
        color=TRACE_COLOR,
        marker="o",
        ls="",
        alpha=TRANSPARENCY,
        zorder=-1,
    )

    bax.axhline(t1_avg_us, ls="--", color=TRACE_COLOR)
    ax.set_ylim(*bax.get_ylim()[0])
    ax.set_yticks([200, 400, 600])
    ax.axhline(t1_avg_us, ls="--", color=TRACE_COLOR, zorder=1)

    bax.set_xlabel("Time (day)", labelpad=40)
    bax.set_ylabel(r"$\mathrm{T_1}$ (μs)", labelpad=60)
    bax.set_yticks([200, 400, 600])
    bax.axs[0].set_xticks([0, 1, 2])
    bax.axs[1].set_xticks([3, 4, 5, 6, 7])
    bax.axs[2].set_xticks([11, 12])

    bax.axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(100))

    secax = bax.secondary_yaxis(
        functions=Q_functions, label=r"$Q$ (million)", labelpad=45
    )

    bax.axs[1].text(
        0.5,
        0.95,
        t1_avg_str,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
