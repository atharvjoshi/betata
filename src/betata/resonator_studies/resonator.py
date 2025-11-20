""" """

from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd

DATA_FOLDER = Path(__file__).parents[3] / "data/resonator_studies"
SPR_SIM_FILEPATH = DATA_FOLDER / "spr_sim.csv"

@dataclass
class Resonator:
    """ """

    name: str
    type: str
    design_name: str
    cooldown_name: str
    film_thickness: float

    # for cpw, pitch is the distance between center conductor and ground plane
    # for le, pitch is the distance between capacitor pads
    pitch: float

    # for cpw, length and width are those of the center conductor
    # for le, length and width are those of the inductor
    length: float = None
    width: float = None

    # simulated resonance frequency and inductance assuming only geometric inductance
    fr_geom: float = None
    l_geom: float = None

    # simulated surface participation ratios
    p_ms: float = None # metal-substrate
    p_ma: float = None # metal-air
    p_sa: float = None # substrate-air
    p_sub: float = None # substrate

    # total attenuation in the input measurement chain
    line_attenuation: float = None

def save_resonator(resonator: Resonator, filepath: Path):
    """ """
    with h5py.File(filepath, "a") as file:
        for key, value in resonator.__dict__.items():
            file.attrs[key] = value

def add_spr_metadata(resonator: Resonator):
    """ """
    df = pd.read_csv(SPR_SIM_FILEPATH)
    pitch_um = round(resonator.pitch * 1e6)
    metadata_row = df.loc[df["pitch (um)"] == pitch_um]
    resonator.width, = metadata_row["width (um)"] * 1e-6
    resonator.p_ms, = metadata_row["p_ms"]
    resonator.p_ma, = metadata_row["p_ma"]
    resonator.p_sa, = metadata_row["p_sa"]
    resonator.p_sub, = metadata_row["p_sub"]
    return resonator

def add_inductance_metadata(resonator: Resonator, ):
    """ """
    film_id = resonator.name.split("_")[1]
    inductance_sim_filepath = DATA_FOLDER / f"{film_id}_inductance_sim.csv"
    df = pd.read_csv(inductance_sim_filepath)
    pitch_um = round(resonator.pitch * 1e6)
    metadata_row = df.loc[(df["pitch (um)"] == pitch_um) & (df["l_s (pH/sq)"] == 0)]
    resonator.fr_geom, = metadata_row["fr_geom (GHz)"] * 1e9
    resonator.l_geom, = metadata_row["l (nH)"] * 1e-9
    return resonator
