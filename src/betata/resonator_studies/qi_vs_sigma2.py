""" """

from pathlib import Path

import numpy as np
from scipy.constants import h, e, mu_0
import lmfit

from betata import plt, get_blues
from betata.resonator_studies.resonator import load_resonators

PEN_DEPTH = 1.78e-6
TEMP_CUTOFF = 15e-3
ETA_MIN = 1e-3
ETA_MAX = 1e-2

BLUES = get_blues(values=[0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])
TRANSPARENCY = 1.0


def sigma_2(fr):
    """ """
    return 1 / (mu_0 * 2 * np.pi * fr * PEN_DEPTH**2)


def conductive_loss_limit(fr):
    """calculated based on the bulk saturated Lk"""
    prefactor = (h / e**2) * (mu_0 * PEN_DEPTH * 2 * np.pi * fr) ** -1
    upper_limit = prefactor / ETA_MIN
    lower_limit = prefactor / ETA_MAX
    return upper_limit, lower_limit


if __name__ == "__main__":
    """ """

    fig_folder = Path(__file__).parents[3] / "out/resonator_studies"
    figsavepath = fig_folder / "qi_vs_sigma2_slides.svg"

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

    all_sigma_2s = []
    all_low_power_qis = []

    for resonator in resonators:
        low_power_trace = None
        for trace in resonator.traces:
            if trace.id == low_power_qi_trace_ids[resonator.name]:
                low_power_trace = trace

        sigma_2_ = sigma_2(resonator.fr_bare)
        low_power_qi = low_power_trace.Qi
        low_power_qi_err = low_power_trace.Qi_err

        all_sigma_2s.append(sigma_2_)
        all_low_power_qis.append(low_power_qi)

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
            sigma_2_,
            low_power_qi,
            yerr=low_power_qi_err,
            ls="",
            color=color,
            marker="o",
            label="data",
        )

    all_sigma_2s = np.array(all_sigma_2s)
    all_low_power_qis = np.array(all_low_power_qis)

    linear_model = lmfit.models.LinearModel()
    linear_model.set_param_hint("intercept", value=0, vary=False)
    fit_result = linear_model.fit(all_low_power_qis, x=all_sigma_2s)
    print(fit_result.fit_report())

    #fr_dummy = np.linspace(3.5e9, 8e9, 1001)
    #sigma_2_dummy = sigma_2(fr_dummy)
    #conductive_loss_limits = conductive_loss_limit(fr_dummy)
    #upper_loss_limit, lower_loss_limit = conductive_loss_limits
    #ax.plot(sigma_2_dummy, upper_loss_limit)
    #ax.plot(sigma_2_dummy, lower_loss_limit)

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_xlabel(r"$\sigma_2 = (\mu_0 \omega \lambda^2)^{-1}$ (Sm$^{-1}$)")
    ax.set_ylabel(r"$Q_\mathrm{int}$")

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
