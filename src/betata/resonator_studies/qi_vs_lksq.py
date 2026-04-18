""" """

from pathlib import Path

import numpy as np

from betata import plt, get_blues
from betata.resonator_studies.resonator import load_resonators

TEMP_CUTOFF = 15e-3

BLUES = get_blues(values=[0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])
TRANSPARENCY = 1.0

if __name__ == "__main__":
    """ """

    fig_folder = Path(__file__).parents[3] / "out/resonator_studies"
    figsavepath = fig_folder / "qi_vs_lksq.svg"

    resonators = load_resonators()

    low_power_qi_trace_ids = {}

    for resonator in resonators:
        min_power, min_power_trace_id = np.inf, None
        for trace in resonator.traces:
            if trace.temperature < TEMP_CUTOFF and not trace.is_excluded:
                if trace.power < min_power:
                    min_power = trace.power
                    min_power_trace_id = trace.id
        low_power_qi_trace_ids[resonator.name] = min_power_trace_id

    fig, ax = plt.subplots(figsize=(5, 5))

    for resonator in resonators:
        low_power_trace = None
        for trace in resonator.traces:
            if trace.id == low_power_qi_trace_ids[resonator.name]:
                low_power_trace = trace
        
        if resonator.l_sheet is None:
            continue

        l_sheet_pH = resonator.l_sheet * 1e12
        l_sheet_err_pH = resonator.l_sheet_err * 1e12
        low_power_qi = low_power_trace.Qi
        low_power_qi_err = low_power_trace.Qi_err

        thickness = resonator.film_thickness * 1e6
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
            l_sheet_pH,
            low_power_qi,
            xerr=l_sheet_err_pH,
            yerr=low_power_qi_err,
            ls="",
            color=color,
            marker="o",
            label="data",
        )

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_xlabel(r"$L_\mathrm{k/◻}$ (pH/$◻$)")
    ax.set_ylabel(r"$Q_\mathrm{int}$")

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
