""" """

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

DATA_FOLDER = Path(__file__).parents[3] / "data/qubit_measurements"
OUTPUT_FOLDER = Path(__file__).parents[3] / "out/qubit_measurements"


@dataclass
class Qubit:
    """ """

    name: str
    design_name: str

    f_q: float = None
    f_r: float = None
    chi: float = None
    kappa: float = None
    Ej: float = None
    Ec: float = None

    t1: np.ndarray = None
    t1_err: np.ndarray = None
    t1_timestamp: np.ndarray = None
    t1_trace_id: np.ndarray = None
    t1_A: np.ndarray = None
    t1_A_err: np.ndarray = None
    t1_B: np.ndarray = None
    t1_B_err: np.ndarray = None

    t1_avg: float = None
    t1_avg_err: float = None

    t2r_avg: float = None
    t2r_avg_err: float = None

    t2e: np.ndarray = None
    t2e_err: np.ndarray = None
    t2e_timestamp: np.ndarray = None
    t2e_trace_id: np.ndarray = None
    t2e_A: np.ndarray = None
    t2e_A_err: np.ndarray = None
    t2e_B: np.ndarray = None
    t2e_B_err: np.ndarray = None

    t2e_avg: float = None
    t2e_avg_err: float = None

    @property
    def Delta(self):
        if None in [self.f_r, self.f_q]:
            return None
        return np.abs(self.f_r - self.f_q)

    @property
    def q_avg(self):
        """ """
        if None in [self.f_q, self.t1_avg]:
            return None
        return 2 * np.pi * self.f_q * self.t1_avg

    @property
    def q_avg_err(self):
        """ """
        if None in [self.f_q, self.t1_avg_err]:
            return None
        return 2 * np.pi * self.f_q * self.t1_avg_err


def load_qubit(filepath: Path) -> Qubit:
    """ """
    with h5py.File(filepath, "a") as file:
        qubit = Qubit(
            name=file.attrs["name"],
            design_name=file.attrs["design_name"],
            f_q=file.attrs["f_q"],
            f_r=file.attrs["f_r"],
            chi=file.attrs["chi"],
            kappa=file.attrs["kappa"],
            Ej=file.attrs["Ej"],
            Ec=file.attrs["Ec"],
            t1=file["t1"]["t1"][:],
            t1_err=file["t1"]["t1_err"][:],
            t1_timestamp=file["t1"]["t1_timestamp"][:],
            t1_trace_id=file["t1"]["t1_trace_id"][:],
            t1_A=file["t1"]["t1_A"][:],
            t1_A_err=file["t1"]["t1_A_err"][:],
            t1_B=file["t1"]["t1_B"][:],
            t1_B_err=file["t1"]["t1_B_err"][:],
            t1_avg=file.attrs["t1_avg"],
            t1_avg_err=file.attrs["t1_avg_err"],
            t2r_avg=file.attrs["t2r_avg"],
            t2r_avg_err=file.attrs["t2r_avg_err"],
            t2e=file["t2e"]["t2e"][:],
            t2e_err=file["t2e"]["t2e_err"][:],
            t2e_timestamp=file["t2e"]["t2e_timestamp"][:],
            t2e_trace_id=file["t2e"]["t2e_trace_id"][:],
            t2e_A=file["t2e"]["t2e_A"][:],
            t2e_A_err=file["t2e"]["t2e_A_err"][:],
            t2e_B=file["t2e"]["t2e_B"][:],
            t2e_B_err=file["t2e"]["t2e_B_err"][:],
            t2e_avg=file.attrs["t2e_avg"],
            t2e_avg_err=file.attrs["t2e_avg_err"],
        )

    # handle None values
    for k, v in qubit.__dict__.items():
        if isinstance(v, h5py._hl.base.Empty):
            setattr(qubit, k, None)

    return qubit


def load_qubits() -> list[Qubit]:
    """ """
    qubits: list[Qubit] = []

    # each file in the output folder is an hdf5 file storing qubit metadata
    for qubit_file in OUTPUT_FOLDER.iterdir():
        if qubit_file.suffix not in (".h5", ".hdf5"):
            continue

        qubit = load_qubit(qubit_file)
        qubits.append(qubit)
    return qubits


def save_qubit(qubit: Qubit, filepath: Path = None):
    """ """
    t1_arrs = [
        "t1",
        "t1_err",
        "t1_timestamp",
        "t1_trace_id",
        "t1_A",
        "t1_A_err",
        "t1_B",
        "t1_B_err",
    ]

    t2e_arrs = [
        "t2e",
        "t2e_err",
        "t2e_timestamp",
        "t2e_trace_id",
        "t2e_A",
        "t2e_A_err",
        "t2e_B",
        "t2e_B_err",
    ]

    if filepath is None:
        for qubit_file in OUTPUT_FOLDER.iterdir():
            if qubit_file.suffix in [".h5", ".hdf5"] and qubit_file.stem == qubit.name:
                filepath = qubit_file
                break

    with h5py.File(filepath, "a") as file:
        # save attributes
        for key, value in qubit.__dict__.items():
            if key in []:  # ignore these attributes
                pass
            elif key in t1_arrs:  # save T1 arrays
                if value is None:  # create dummy stand-in dataset
                    value = np.zeros(1)

                t1_group = file.require_group("t1")

                if key in t1_group:  # prepare to overwrite dataset
                    del t1_group[key]

                t1_group.create_dataset(key, data=value)
            elif key in t2e_arrs:  # save T2E arrays
                if value is None:  # create dummy stand-in dataset
                    value = np.zeros(1)

                t2e_group = file.require_group("t2e")

                if key in t2e_group:  # prepare to overwrite dataset
                    del t2e_group[key]

                t2e_group.create_dataset(key, data=value)
            else:
                # handle None values
                if value is None:
                    value = h5py.Empty("S10")

                file.attrs[key] = value

        # save properties
        for prop in ["Delta", "q_avg", "q_avg_err"]:
            value = getattr(qubit, prop)

            # handle None values
            if value is None:
                value = h5py.Empty("S10")

            file.attrs[prop] = value
