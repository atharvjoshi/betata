""" """

from dataclasses import dataclass
from pathlib import Path
from operator import attrgetter

import h5py
import numpy as np

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


def load_raw_trace(filepath: Path):
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

def load_fitted_trace(filepath: Path):
    """ """

def load_traces(folder: Path, raw=True):
    """ raw: bool to specify whether to load a raw trace (True) or a fitted trace (False) """
    load_fn = load_raw_trace if raw else load_fitted_trace

    traces = []
    for filepath in folder.iterdir():
        if filepath.suffix in [".h5", ".hdf5"]:
            traces.append(load_fn(filepath))

    # id traces by power (decreasing), then by temperature (increasing)
    def sort_fn(trace):
        return (-attrgetter("power")(trace), attrgetter("temperature")(trace))
    sorted_traces = sorted(traces, key=sort_fn)

    for idx, trace in enumerate(sorted_traces):
        trace.id = idx

    return sorted_traces

def save_traces(traces: list[Trace], filepath: Path):
    """ """
    with h5py.File(filepath, "a") as file:
        for trace in traces:
            trace_group = file.require_group(trace.filename)
            for key, value in trace.__dict__.items():
                if key not in ["frequency", "s21imag", "s21real", "filename"]:
                        
                        #handle None values
                        if value is None:
                            value = h5py.Empty("S10")

                        trace_group.attrs[key] = value
