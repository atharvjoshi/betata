""" """

from pathlib import Path

import h5py
import matplotlib.ticker as ticker
import numpy as np

from betata import plt, get_purples
from betata.qubit_measurements.traces import load_t2r_trace
from betata.qubit_measurements.fit_t2r_traces.fit_t2r_traces import (
    fit_t2r_trace,
    t2r_fit_fn,
)

TRACE_COLOR = "#2AA198"  # get_purples(1, 0.9, 0.9)[0]
TRANSPARENCY = 0.85

if __name__ == "__main__":
    """ """

    qubit_name = "Q3_2p73"

    qubit_metadata_folder = Path(__file__).parents[4] / "out/qubit_measurements"
    trace_data_folder = Path(__file__).parents[4] / "data/qubit_measurements"
    qubit_trace_data_folder = trace_data_folder / f"{qubit_name}/T2R_{qubit_name}"

    figsavepath = qubit_metadata_folder / f"{qubit_name}_T2R_max.svg"

    max_t2r_trace_id = None

    # find trace id with highest T2R
    for filepath in qubit_metadata_folder.iterdir():
        if filepath.suffix not in [".h5", ".hdf5"]:
            continue
        with h5py.File(filepath) as file:
            if qubit_name != file.attrs["name"]:
                continue
            all_t2rs = file["t2r"]["t2r"][:]
            all_t2r_trace_ids = file["t2r"]["t2r_trace_id"][:]
            max_t2r_trace_id = all_t2r_trace_ids[np.argmax(all_t2rs)]

    # max_t2r_trace_id = 773 # Q7_4p83

    trace_filepath = None
    for filepath in qubit_trace_data_folder.iterdir():
        if filepath.suffix not in [".h5", ".hdf5"]:
            continue
        with h5py.File(filepath) as file:
            if file.attrs["id"] == max_t2r_trace_id:
                trace_filepath = filepath

    trace = load_t2r_trace(trace_filepath)
    fit_result = fit_t2r_trace(trace, plot=False)

    print(fit_result.fit_report())

    tau_ms = trace.tau * 1e3
    tau_ms_dummy = np.linspace(min(tau_ms), max(tau_ms), 1001)

    # downsample if needed
    tau_ms = tau_ms
    corrected_population = trace.population

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        tau_ms,
        corrected_population,
        color=TRACE_COLOR,
        alpha=TRANSPARENCY,
        zorder=-1,
    )

    T2R_us = trace.T2R * 1e6
    T2R_err_us = trace.T2R_err * 1e6
    T2R_str = r"$\mathrm{T_{2, R}}$" + f" = {T2R_us:.0f} ± {T2R_err_us:.0f} μs"

    best_fit = t2r_fit_fn(tau_ms_dummy * 1e-3, **fit_result.params)
    ax.plot(tau_ms_dummy, best_fit, color=TRACE_COLOR, alpha=0.5, lw=2)

    ax.set_xlabel(r"$\tau$ (ms)")
    ax.set_ylabel(r"$\mathrm{P_e}$")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.set_ylim(-0.05, 1.05)

    ax.text(
        0.95,
        0.95,
        T2R_str,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        color=TRACE_COLOR,
    )

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
