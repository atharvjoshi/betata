""" """

from pathlib import Path

import h5py
import matplotlib.ticker as ticker
import numpy as np

from betata import plt
from betata.qubit_measurements.qubit import load_qubit
from betata.qubit_measurements.traces import load_t1_trace, load_t2e_trace
from betata.qubit_measurements.fit_t1_traces.fit_t1_traces import (
    fit_t1_trace,
    t1_fit_fn,
)
from betata.qubit_measurements.fit_t2e_traces.fit_t2e_traces import (
    fit_t2e_trace,
    t2e_fit_fn,
)

T1_TRACE_COLOR = "#E77500"
T2E_TRACE_COLOR = "#003D7C"
TRANSPARENCY = 0.85

if __name__ == "__main__":
    """ """

    qubit_name = "Q6_4p69"

    out_folder = Path(__file__).parents[3] / "out"
    qubit_file = out_folder / f"qubit_measurements/{qubit_name}.h5"
    qubit = load_qubit(qubit_file)

    best_t1_trace_id = qubit.t1_trace_id[np.argmax(qubit.t1)]
    best_t2e_trace_id = qubit.t2e_trace_id[np.argmax(qubit.t2e)]

    figsavepath = out_folder / f"qubit_measurements/{qubit_name}_T1_T2E_max_fig4.png"

    trace_data_folder = Path(__file__).parents[3] / "data/qubit_measurements"
    t1_trace_data_folder = trace_data_folder / f"{qubit_name}/T1_{qubit_name}"
    t2e_trace_data_folder = trace_data_folder / f"{qubit_name}/T2E_{qubit_name}"

    t1_trace_filepath = None
    for filepath in t1_trace_data_folder.iterdir():
        if filepath.suffix not in [".h5", ".hdf5"]:
            continue
        with h5py.File(filepath) as file:
            if file.attrs["id"] == best_t1_trace_id:
                t1_trace_filepath = filepath

    t2e_trace_filepath = None
    for filepath in t2e_trace_data_folder.iterdir():
        if filepath.suffix not in [".h5", ".hdf5"]:
            continue
        with h5py.File(filepath) as file:
            if file.attrs["id"] == best_t2e_trace_id:
                t2e_trace_filepath = filepath


    t1_trace = load_t1_trace(t1_trace_filepath)
    t1_fit_result = fit_t1_trace(t1_trace, plot=False)

    t2e_trace = load_t2e_trace(t2e_trace_filepath)
    t2e_fit_result = fit_t2e_trace(t2e_trace, plot=False)

    t1_tau_ms = t1_trace.tau * 1e3
    t1_tau_ms_dummy = np.linspace(min(t1_tau_ms), max(t1_tau_ms), 1001)

    t2e_tau_ms = t2e_trace.tau * 1e3
    t2e_tau_ms_dummy = np.linspace(min(t2e_tau_ms), max(t2e_tau_ms), 1001)

    # correct for constant offset B
    corrected_t1_population = t1_trace.population - t1_trace.B
    corrected_t2e_population = t2e_trace.population - t2e_trace.B

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        t1_tau_ms,
        corrected_t1_population,
        color=T1_TRACE_COLOR,
        zorder=-1,
        alpha=TRANSPARENCY,
    )

    T1_us = t1_trace.T1 * 1e6
    T1_err_us = t1_trace.T1_err * 1e6
    T1_str = r"$\mathrm{T_1}$" + f" = {T1_us:.0f} ± {T1_err_us:.0f} μs"

    ax.scatter(
        t2e_tau_ms,
        corrected_t2e_population,
        color=T2E_TRACE_COLOR,
        zorder=-1,
        alpha=TRANSPARENCY,
    )

    T2E_us = t2e_trace.T2E * 1e6
    T2E_err_us = t2e_trace.T2E_err * 1e6
    T2E_str = r"$\mathrm{T_{2, E}}$" + f" = {T2E_us:.0f} ± {T2E_err_us:.0f} μs"

    t1_best_fit = t1_fit_fn(t1_tau_ms_dummy, t1_trace.A, T1_us * 1e-3, 0)
    ax.plot(t1_tau_ms_dummy, t1_best_fit, color=T1_TRACE_COLOR)

    t2e_best_fit = t2e_fit_fn(t2e_tau_ms_dummy, t2e_trace.A, T2E_us * 1e-3, 0)
    ax.plot(t2e_tau_ms_dummy, t2e_best_fit, color=T2E_TRACE_COLOR)

    ax.set_xlabel(r"$\tau$ (ms)")
    ax.set_ylabel(r"$\mathrm{P_e}$")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax.text(
        0.65,
        0.2,
        T1_str,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        color=T1_TRACE_COLOR,
    )

    ax.text(
        0.65,
        0.6,
        T2E_str,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        color=T2E_TRACE_COLOR,
    )

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()