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

    t1_avg: float = None
    t1_avg_err: float = None
    t2r_avg: float = None
    t2r_avg_err: float = None
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
            alpha=file.attrs["alpha"],
            Ej=file.attrs["Ej"],
            Ec=file.attrs["Ec"],
            t1_avg=file.attrs["t1_avg"],
            t1_avg_err=file.attrs["t1_avg_err"],
            t2r_avg=file.attrs["t2r_avg"],
            t2r_avg_err=file.attrs["t2r_avg_err"],
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

    if filepath is None:
        for qubit_file in OUTPUT_FOLDER.iterdir():
            if qubit_file.stem == qubit.name:
                filepath = qubit_file
                break

    with h5py.File(filepath, "a") as file:
        # save attributes
        for key, value in qubit.__dict__.items():
            # ignore these attributes
            if key in []:
                continue

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
