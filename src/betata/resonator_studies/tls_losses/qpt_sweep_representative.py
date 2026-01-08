"""Show a representative Qi vs power, temp sweep result as a subfigure"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from rrfit.fitfns import dBmtoW
from rrfit.waterfall import QIntVsTemp_consistent

from betata import plt, get_purples
from betata.resonator_studies.resonator import Resonator, load_resonator
from betata.resonator_studies.trace import Trace, load_fitted_traces

TRANSPARENCY = 0.85


def get_qint_fit(resonator, temps, frs, qints, power, qc):
    """ """
    num_temps = 101
    temps_dummy = np.linspace(min(temps), max(temps), num_temps)
    frs_dummy = np.interp(temps_dummy, temps, frs)
    qint_dummy = np.interp(temps_dummy, temps, qints)

    fit_params = {
        "delta_QP0": resonator.qpt_fit_params["delta_QP0"]["value"],
        "Q_TLS0": resonator.qpt_fit_params["Q_TLS0"]["value"],
        "D_0": resonator.qpt_fit_params["D_0"]["value"],
        "tc": resonator.qpt_fit_params["tc"]["value"],
        "Q_other": resonator.qpt_fit_params["Q_other"]["value"],
        "beta": resonator.qpt_fit_params["beta"]["value"],
        "beta2": resonator.qpt_fit_params["beta2"]["value"],
    }

    powers = np.ones(num_temps) * dBmtoW(power - resonator.line_attenuation)

    qint_fit = QIntVsTemp_consistent(
        temps_dummy,
        fit_params,
        frs_dummy,
        powers,
        qc,
        qint_dummy,
    )

    return temps_dummy, np.array(qint_fit)


if __name__ == "__main__":
    """ """

    resonator_name = "R70_F11_5p59"
    resonator_folder = Path(__file__).parents[4] / "out/resonator_studies"
    resonator_file = resonator_folder / f"{resonator_name}.h5"

    figsavepath = resonator_folder / "power_temp_sweep.png"

    resonator: Resonator = load_resonator(resonator_file)
    fitted_traces: list[Trace] = load_fitted_traces(resonator_file)

    data = defaultdict(dict)  # key: power, value: dict with key=temperature value=Trace
    # exclude -20 dBm points to de-clutter the figure
    for trace in fitted_traces:
        if trace.id in resonator.qpt_fit_trace_ids and trace.power != -20:
            data[trace.power][trace.temperature] = trace

    fig, ax = plt.subplots(figsize=(9, 6))
    num_series = len(data.keys())
    purples = get_purples(num_series, start=0.40, stop=1.00)
    powers = []
    for idx, (power, inner_dict) in enumerate(data.items()):
        powers.append(power)
        temps = np.array(list(inner_dict.keys()))
        traces = list(inner_dict.values())
        frs = np.array([trace.fr for trace in traces])
        qints = np.array([trace.Qi for trace in traces])
        qint_errs = np.array([trace.Qi_err for trace in traces])
        qls = np.array([trace.Ql for trace in traces])
        qc = np.mean(np.array([trace.absQc for trace in traces]))

        temps_mK = temps * 1e3

        temps_dummy, qint_fit = get_qint_fit(resonator, temps, frs, qints, power, qc)
        temps_dummy_mK = temps_dummy * 1e3

        ax.errorbar(
            temps_mK,
            qints,
            yerr=qint_errs,
            ls="",
            color=purples[idx],
            marker="o",
            label=f"{power} dBm",
            alpha=TRANSPARENCY,
        )

        ax.plot(temps_dummy_mK, qint_fit, color=purples[idx])

    ax.set_xlabel("Temperature (mK)")
    ax.set_ylabel(r"$Q_{\mathrm{int}}$")

    ax.set_yscale("log")
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_xticks([10, 30, 50, 70, 90, 110, 130], minor=True)

    # colorbar legend
    cbaxes = inset_axes(ax, width="20%", height="4%", loc="upper right", borderpad=2)
    purple_cmap = colors.ListedColormap(purples).reversed()
    purple_norm = colors.Normalize(vmin=-95, vmax=-25)
    sm = ScalarMappable(cmap=purple_cmap, norm=purple_norm)
    cbar = fig.colorbar(sm, cax=cbaxes, orientation="horizontal")
    cbar.set_ticks([-90, -30])
    cbar.ax.set_title("Power (dBm)")

    # ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.2, 0.7))

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")
    plt.show()
