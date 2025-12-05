"""Calculate kinetic inductance fraction alpha from bare resonator frequency shifts"""

from pathlib import Path

import numpy as np

from betata.resonator_studies.trace import load_fitted_traces
from betata.resonator_studies.resonator import Resonator, load_resonator, save_resonator


def find_alpha_bare(fr_geom: float, fr_bare: float) -> float:
    """ """
    return 1 - (fr_bare / fr_geom) ** 2


def find_fr_bare(resonator: Resonator) -> float | None:
    """
    fr_bare is operationalized as the average of the fitted resonance frequency of the non-excluded S21 traces below 20 mK.
    """
    TEMP_CUTOFF = 20e-3
    # traces are pre-sorted by power (decreasing) and temperature (increasing)
    frs_bare = []
    for trace in resonator.traces:
        if not trace.is_excluded and trace.temperature < TEMP_CUTOFF:
            frs_bare.append(trace.fr)
    return None if not frs_bare else np.mean(np.array(frs_bare))


if __name__ == "__main__":
    """ """

    resonators: list[Resonator] = []

    # each file in the output folder is an hdf5 file storing resonator metadata
    output_folder = Path(__file__).parents[4] / "out/resonator_studies"
    for resonator_file in output_folder.iterdir():
        if resonator_file.suffix not in (".h5", ".hdf5"):
            continue

        resonator = load_resonator(resonator_file)
        resonator.traces = load_fitted_traces(resonator_file)
        resonators.append(resonator)
        resonator.fr_bare = find_fr_bare(resonator)
        resonator.alpha_bare = find_alpha_bare(resonator.fr_geom, resonator.fr_bare)
    
        save_resonator(resonator, resonator_file)
