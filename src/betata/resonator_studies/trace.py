""" """

from dataclasses import dataclass
from pathlib import Path
from operator import attrgetter

import h5py
import numpy as np

@dataclass
class Trace:
    """ """

    filename: str
    resonator_name: str
    frequency: np.ndarray
    s21imag: np.ndarray
    s21real: np.ndarray
    temperature: float
    temperature_err: float
    power: float
    tau: float

    id: int = None
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

def load_traces(folder: Path):
    """ """
    traces = []
    for filepath in folder.iterdir():
        if filepath.suffix in [".h5", ".hdf5"]:
            traces.append(load_trace(filepath))
    # id traces by power, then by temperature
    traces.sort(key=attrgetter("power", "temperature"))
    for idx, trace in enumerate(traces):
        trace.id = idx
    return traces

def save_traces(traces: list[Trace], filepath: Path):
    """ """
    with h5py.File(filepath, "a") as file:
        for trace in traces:
            trace_group = file.require_group(trace.filename)
            for key, value in trace.__dict__.items():
                if key not in ["frequency", "s21imag", "s21real", "filename"]:
                    trace_group.attrs[key] = value
