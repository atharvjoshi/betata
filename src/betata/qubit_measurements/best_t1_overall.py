"""Generate figure of the best T1 trace"""

from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np

from betata import plt, get_purples
from betata.qubit_measurements.traces import load_t1_trace
from betata.qubit_measurements.fit_t1_traces.fit_t1_traces import (
    fit_t1_trace,
    t1_fit_fn,
)

TRACE_COLOR = get_purples(1, 1.0, 1.0)[0]
TRANSPARENCY = 0.85

TRACE_FILEPATH = (
    Path(__file__).parents[3]
    / "data/qubit_measurements/Q3_2p73/T1_Q3_2p73/2026-01-17_12-03-04_Q3_2p73_T1.h5"
)
# this confusion matrix is obtained from a single shot calibration measurement acquired right before the trace above
CONFUSION_MATRIX = np.array([[0.878, 0.196], [0.122, 0.804]])

if __name__ == "__main__":
    """ """

    figsavepath = Path(__file__).parents[3] / "out/qubit_measurements/t1_max.svg"

    cprime = np.linalg.inv(CONFUSION_MATRIX)

    trace = load_t1_trace(TRACE_FILEPATH)
    population = trace.population
    corrected_population = population * cprime[1][0] + population * cprime[1][1]
    trace.population = corrected_population
    fit_result = fit_t1_trace(trace, plot=False)

    tau_ms = trace.tau * 1e3
    tau_ms_dummy = np.linspace(min(tau_ms), max(tau_ms), 1001)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        tau_ms,
        trace.population,
        color=TRACE_COLOR,
        alpha=TRANSPARENCY,
        zorder=-1,
    )

    T1_us = trace.T1 * 1e6
    T1_err_us = trace.T1_err * 1e6
    T1_str = r"$\mathrm{T_1}$" + f" = {T1_us:.0f} ± {T1_err_us:.0f} μs"

    best_fit = t1_fit_fn(tau_ms_dummy, trace.A, T1_us * 1e-3, trace.B)
    ax.plot(tau_ms_dummy, best_fit, color=TRACE_COLOR)

    ax.set_xlabel(r"$\tau$ (ms)")
    ax.set_ylabel(r"$\mathrm{P_e}$")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_ylim(-0.05, 1.05)

    ax.text(
        0.95,
        0.95,
        T1_str,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")
    plt.show()
