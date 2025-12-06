""" """

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import lmfit
import pandas as pd

from betata.resonator_studies.trace import Trace

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

    # bare resonance frequency, measured at highest power and lowest temperature
    fr_bare: float = None

    # kinetic inductance fraction alpha calculated from bare frequency shifts
    alpha_bare: float = None

    # simulated surface participation ratios
    p_ms: float = None  # metal-substrate
    p_ma: float = None  # metal-air
    p_sa: float = None  # substrate-air
    p_sub: float = None  # substrate

    # total attenuation in the input measurement chain
    line_attenuation: float = None

    # list of fitted traces
    traces: list[Trace] = None

    # fit parameters from the Q_int vs P, T sweep
    qpt_fit_params: dict = None  # stored as a json string

    # fit parameters from the ffs vs T sweep
    ffs_fit_params: dict = None  # stored as a json string

    # trace ids included in fits
    qpt_fit_trace_ids: list[int] = None
    ffs_fit_trace_ids: list[int] = None


def load_resonator(filepath: Path) -> Resonator:
    """ """
    with h5py.File(filepath, "a") as file:
        resonator = Resonator(
            name=file.attrs["name"],
            type=file.attrs["type"],
            design_name=file.attrs["design_name"],
            cooldown_name=file.attrs["cooldown_name"],
            film_thickness=file.attrs["film_thickness"],
            pitch=file.attrs["pitch"],
            length=file.attrs.get("length"),
            width=file.attrs.get("width"),
            fr_geom=file.attrs.get("fr_geom"),
            l_geom=file.attrs.get("l_geom"),
            fr_bare=file.attrs.get("fr_bare"),
            alpha_bare=file.attrs.get("alpha_bare"),
            p_ms=file.attrs.get("p_ms"),
            p_ma=file.attrs.get("p_ma"),
            p_sa=file.attrs.get("p_sa"),
            p_sub=file.attrs.get("p_sub"),
            line_attenuation=file.attrs.get("line_attenuation"),
            traces=file.attrs.get("traces"),
            qpt_fit_params=file.attrs.get("qpt_fit_params"),
            ffs_fit_params=file.attrs.get("ffs_fit_params"),
            qpt_fit_trace_ids=file.attrs.get("qpt_fit_trace_ids"),
            ffs_fit_trace_ids=file.attrs.get("ffs_fit_trace_ids"),
        )

    # handle None values
    for k, v in resonator.__dict__.items():
        if isinstance(v, h5py._hl.base.Empty):
            setattr(resonator, k, None)

    # convert json strings to dicts
    if resonator.qpt_fit_params is not None:
        resonator.qpt_fit_params = json.loads(resonator.qpt_fit_params)
    if resonator.ffs_fit_params is not None:
        resonator.ffs_fit_params = json.loads(resonator.ffs_fit_params)

    return resonator


def save_resonator(resonator: Resonator, filepath: Path):
    """ """
    with h5py.File(filepath, "a") as file:
        for key, value in resonator.__dict__.items():
            # ignore these attributes
            if key in ["traces"]:
                continue

            # handle None values
            if value is None:
                value = h5py.Empty("S10")

            # handle dict values -> convert to json string
            if isinstance(value, dict):
                value = json.dumps(value)

            file.attrs[key] = value


def add_spr_metadata(resonator: Resonator):
    """ """
    df = pd.read_csv(SPR_SIM_FILEPATH)
    pitch_um = round(resonator.pitch * 1e6)
    metadata_row = df.loc[df["pitch (um)"] == pitch_um]
    (resonator.width,) = metadata_row["width (um)"] * 1e-6
    (resonator.p_ms,) = metadata_row["p_ms"]
    (resonator.p_ma,) = metadata_row["p_ma"]
    (resonator.p_sa,) = metadata_row["p_sa"]
    (resonator.p_sub,) = metadata_row["p_sub"]
    return resonator


def add_inductance_metadata(resonator: Resonator):
    """ """
    lk_sim_filepath = DATA_FOLDER / f"{resonator.design_name}_lk_sim.csv"
    df = pd.read_csv(lk_sim_filepath)
    pitch_um = round(resonator.pitch * 1e6)
    metadata_row = df.loc[(df["pitch (um)"] == pitch_um) & (df["l_s (pH/sq)"] == 0)]
    (resonator.fr_geom,) = metadata_row["fr_geom (GHz)"] * 1e9
    (resonator.l_geom,) = metadata_row["l (nH)"] * 1e-9
    return resonator


def add_qpt_fit_params(resonator: Resonator, fit_params: lmfit.Parameters):
    """ """
    fit_params_dict = {}
    for param in fit_params.values():
        param: lmfit.Parameter
        fit_params_dict[param.name] = {}
        fit_params_dict[param.name]["value"] = param.value
        fit_params_dict[param.name]["stderr"] = param.stderr
        fit_params_dict[param.name]["correl"] = param.correl
    resonator.qpt_fit_params = fit_params_dict

    # remove temporary attributes added by waterfall.py, if present
    unwanted_attrs = ("best_params", "waterfall_fit_result")
    for attr in unwanted_attrs:
        if hasattr(resonator, attr):
            delattr(resonator, attr)


def add_ffs_fit_params(resonator: Resonator, fit_params: lmfit.Parameters):
    """ """
