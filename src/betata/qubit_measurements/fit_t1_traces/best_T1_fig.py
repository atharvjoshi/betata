"""Generate figures of the best T1 traces for each qubit"""

from pathlib import Path

import h5py
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

QUBITS_TO_INCLUDE = {
    "Q1_2p61",
    "Q2_2p80",
    "Q3_2p88",
    "Q4_3p02",
    "Q5_3p19",
    "Q6_4p69",
    "Q7_4p83",
    "Q8_4p93",
    "Q9_5p15",
    "Q10_5p36",
    "Q11_5p78",
}

if __name__ == "__main__":
    """ """

    qubit_metadata_folder = Path(__file__).parents[4] / "out/qubit_measurements"
    trace_data_folder = Path(__file__).parents[4] / "data/qubit_measurements"

    # dict mapping qubit names and best t1 trace ids
    qubit_traceid_dict = {}

    for filepath in qubit_metadata_folder.iterdir():
        if filepath.suffix not in [".h5", ".hdf5"]:
            continue
        with h5py.File(filepath) as file:
            qubit_name = file.attrs["name"]
            all_t1s = file["t1"]["t1"][:]
            all_t1_trace_ids = file["t1"]["t1_trace_id"][:]
            qubit_traceid_dict[qubit_name] = all_t1_trace_ids[np.argmax(all_t1s)]

    # retrieve best T1 trace data and generate best T1 figure for each qubit
    for qubit_name, trace_id in qubit_traceid_dict.items():
        if qubit_name not in QUBITS_TO_INCLUDE:
            continue
    
        qubit_trace_data_folder = trace_data_folder / f"{qubit_name}/T1_{qubit_name}"

        figsavepath = qubit_metadata_folder / f"{qubit_name}_T1_max.png"

        trace_filepath = None
        for filepath in qubit_trace_data_folder.iterdir():
            if filepath.suffix not in [".h5", ".hdf5"]:
                continue
            with h5py.File(filepath) as file:
                if file.attrs["id"] == trace_id:
                    trace_filepath = filepath

        trace = load_t1_trace(trace_filepath)
        fit_result = fit_t1_trace(trace, plot=False)

        tau_ms = trace.tau * 1e3
        tau_ms_dummy = np.linspace(min(tau_ms), max(tau_ms), 1001)

        # correct for constant offset B
        corrected_population = trace.population - trace.B

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            tau_ms,
            corrected_population,
            color=TRACE_COLOR,
            alpha=TRANSPARENCY,
            zorder=-1,
        )

        T1_us = trace.T1 * 1e6
        T1_err_us = trace.T1_err * 1e6
        T1_str = r"$\mathrm{T_1}$" + f" = {T1_us:.0f} ± {T1_err_us:.0f} μs"

        best_fit = t1_fit_fn(tau_ms_dummy, trace.A, T1_us * 1e-3, 0)
        ax.plot(tau_ms_dummy, best_fit, color=TRACE_COLOR)

        ax.set_xlabel(r"$\tau$ (ms)")
        ax.set_ylabel(r"$\mathrm{P_e}$")

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.set_yscale("log")
        ax.set_yticks([0.01, 0.1, 1.0])
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

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
