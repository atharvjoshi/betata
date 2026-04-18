"""Generate figure of the best T2E trace"""

from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np

from betata import plt, get_purples
from betata.qubit_measurements.traces import load_t2e_trace
from betata.qubit_measurements.fit_t2e_traces.fit_t2e_traces import (
    fit_t2e_trace,
    t2e_fit_fn,
)

TRACE_COLOR = get_purples(1, 1.0, 1.0)[0]
TRANSPARENCY = 0.85

TRACE_FILEPATH = (
    Path(__file__).parents[3]
    / "data/qubit_measurements/Q3_2p73/T2E_Q3_2p73/2026-01-19_21-18-29_Q3_2p73_T2E.h5"
)

if __name__ == "__main__":
    """ """
    
    figsavepath = Path(__file__).parents[3] / "out/qubit_measurements/t2e_max.svg"

    trace = load_t2e_trace(TRACE_FILEPATH)
    fit_result = fit_t2e_trace(trace, plot=False)

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

    T2E_us = trace.T2E * 1e6
    T2E_err_us = trace.T2E_err * 1e6
    T2E_str = r"$\mathrm{T_{2, E}}$" + f" = {T2E_us:.0f} ± {T2E_err_us:.0f} μs"

    best_fit = t2e_fit_fn(tau_ms_dummy, trace.A, T2E_us * 1e-3, trace.B)
    ax.plot(tau_ms_dummy, best_fit, color=TRACE_COLOR)

    ax.set_xlabel(r"$\tau$ (ms)")
    ax.set_ylabel(r"$\mathrm{P_e}$")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_ylim(-0.05, 0.65)

    ax.text(
        0.95,
        0.95,
        T2E_str,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")
    plt.show()
