""" """

from dataclasses import dataclass
from pathlib import Path
from operator import attrgetter

import h5py
import numpy as np
from rrfit.plotfns import plot_hangerfit


@dataclass
class Trace:
    """ """

    # attributes common to all traces
    filename: str
    resonator_name: str
    id: int = None

    # attributes of raw data traces
    frequency: np.ndarray = None
    s21imag: np.ndarray = None
    s21real: np.ndarray = None
    temperature: float = None
    temperature_err: float = None
    power: float = None
    tau: float = None

    # attributes of fitted traces
    background_amp: float = None
    background_phase: float = None
    fr: float = None
    fr_err: float = None
    Qi: float = None
    Qi_err: float = None
    Ql: float = None
    Ql_err: float = None
    absQc: float = None
    absQc_err: float = None
    phi: float = None
    phi_err: float = None
    is_excluded: bool = None


def load_trace(filepath: Path):
    """ """
    with h5py.File(filepath) as file:
        trace = Trace(
            filename=filepath.stem,
            resonator_name=file.attrs["resonator_name"],
            frequency=file["frequency"][:],
            s21imag=file["s21imag"][:],
            s21real=file["s21real"][:],
            temperature=np.mean(file["temperature"][:]),
            temperature_err=np.std(file["temperature"][:]),
            power=file.attrs["power"],
            tau=file.attrs["tau"],
        )
    return trace


def sort_traces_pt(traces: list[Trace]):
    """sort traces by power (decreasing), then by temperature (increasing)"""

    def sort_fn(trace):
        """ """
        return (-attrgetter("power")(trace), attrgetter("temperature")(trace))

    return sorted(traces, key=sort_fn)


def load_traces(folder: Path):
    """ """
    traces = []
    for filepath in folder.iterdir():
        if filepath.suffix in [".h5", ".hdf5"]:
            traces.append(load_trace(filepath))

    # id traces by power (decreasing), then by temperature (increasing)
    sorted_traces = sort_traces_pt(traces)
    for idx, trace in enumerate(sorted_traces):
        trace.id = idx

    return sorted_traces


def load_fitted_traces(filepath: Path):
    """ """
    traces = []
    with h5py.File(filepath) as file:
        for trace_name in file.keys():
            trace_data = file[trace_name]
            trace = Trace(
                filename=trace_name,
                resonator_name=trace_data.attrs["resonator_name"],
                id=trace_data.attrs["id"],
                temperature=trace_data.attrs.get("temperature"),
                temperature_err=trace_data.attrs.get("temperature_err"),
                power=trace_data.attrs.get("power"),
                tau=trace_data.attrs.get("tau"),
                background_amp=trace_data.attrs.get("background_amp"),
                background_phase=trace_data.attrs.get("background_phase"),
                fr=trace_data.attrs.get("fr"),
                fr_err=trace_data.attrs.get("fr_err"),
                Qi=trace_data.attrs.get("Qi"),
                Qi_err=trace_data.attrs.get("Qi_err"),
                Ql=trace_data.attrs.get("Ql"),
                Ql_err=trace_data.attrs.get("Ql_err"),
                absQc=trace_data.attrs.get("absQc"),
                absQc_err=trace_data.attrs.get("absQc_err"),
                phi=trace_data.attrs.get("phi"),
                phi_err=trace_data.attrs.get("phi_err"),
                is_excluded=trace_data.attrs.get("is_excluded"),
            )
            traces.append(trace)
    return sort_traces_pt(traces)


def save_traces(traces: list[Trace], filepath: Path):
    """ """
    with h5py.File(filepath, "a") as file:
        for trace in traces:
            trace_group = file.require_group(trace.filename)
            for key, value in trace.__dict__.items():
                if key not in ["frequency", "s21imag", "s21real", "filename"]:
                    # handle None values
                    if value is None:
                        value = h5py.Empty("S10")

                    trace_group.attrs[key] = value


def plot_fitted_trace(trace: Trace, resonator_name: str):
    """ """
    folder = Path(__file__).parents[3] / f"data/resonator_studies/{resonator_name}"
    for filepath in folder.iterdir():
        if filepath.stem == trace.filename:
            with h5py.File(filepath) as file:
                trace.frequency = file["frequency"][:]
                trace.s21imag = file["s21imag"][:]
                trace.s21real = file["s21real"][:]

    plot_title = f"Device {trace.resonator_name}, Trace #{trace.id}, Power {trace.power} dBm, Temp {trace.temperature * 1e3:.1f}mK"

    plot_hangerfit(trace, plot_title=plot_title)
