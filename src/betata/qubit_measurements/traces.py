""" """

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


@dataclass
class RPMTrace:
    """ """

    qubit_name: str
    qubit_frequency: float
    repetitions: int
    amplitude: np.ndarray
    I_g: np.ndarray
    Q_g: np.ndarray
    I_e: np.ndarray
    Q_e: np.ndarray


def load_rpm_trace(filepath: Path) -> RPMTrace:
    """ """
    with h5py.File(filepath) as file:
        rpm_trace = RPMTrace(
            qubit_name=file.attrs["qubit_name"],
            qubit_frequency=file.attrs["qubit_frequency"],
            repetitions=file.attrs["repetitions"],
            amplitude=file["amplitude"][:],
            I_g=file["I_g"][:],
            Q_g=file["Q_g"][:],
            I_e=file["I_e"][:],
            Q_e=file["Q_e"][:],
        )
    return rpm_trace


@dataclass
class T1Trace:
    """ """

    qubit_name: str
    qubit_frequency: float
    readout_frequency: float
    repetitions: int
    pi_pulse_amplitude: float
    pi_pulse_length: int
    readout_pulse_amplitude: float
    readout_pulse_length: int

    timestamp: datetime
    tau: np.ndarray
    population: np.ndarray

    id: int = None


def load_t1_trace(filepath: Path) -> T1Trace:
    """ """
    with h5py.File(filepath) as file:
        t1_trace = T1Trace(
            qubit_name=file.attrs["qubit_name"],
            qubit_frequency=file.attrs["qubit_frequency"],
            readout_frequency=file.attrs["readout_frequency"],
            repetitions=file.attrs["repetitions"],
            pi_pulse_amplitude=file.attrs["pi_pulse_amplitude"],
            pi_pulse_length=file.attrs["pi_pulse_length"],
            readout_pulse_amplitude=file.attrs["readout_pulse_amplitude"],
            readout_pulse_length=file.attrs["readout_pulse_length"],
            timestamp=datetime.strptime(file.attrs["timestamp"], "%Y-%m-%d %H:%M:%S"),
            tau=file["tau"][:],
            population=file["population"][:],
        )
    return t1_trace


def load_t1_traces(folder: Path) -> list[T1Trace]:
    """ """
    traces: list[T1Trace] = []
    for filepath in folder.iterdir():
        if filepath.suffix in [".h5", ".hdf5"]:
            traces.append(load_t1_trace(filepath))

    # id traces by timestamp (increasing)
    sorted_traces = sorted(traces, key=lambda trace: trace.timestamp)
    for idx, trace in enumerate(sorted_traces):
        trace.id = idx

    return sorted_traces
