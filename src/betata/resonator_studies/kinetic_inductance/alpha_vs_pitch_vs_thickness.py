""" """

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from betata import plt

def get_alpha(fr_meas, fr_geom):
    """ """
    return 1 - (fr_meas / fr_geom) ** 2

if __name__ == "__main__":
    """ """
    
    input_folder = Path(__file__).parents[4] / "out/resonator_studies"

    data = defaultdict(dict)

    for filepath in input_folder.iterdir():

        if filepath.suffix not in [".h5", ".hdf5"]:
            continue

        with h5py.File(filepath) as file:

            if file.attrs["type"] == "LE":
                continue

            if file.attrs["design_name"] == "TAHPLEKI":
                continue

            pitch = file.attrs["pitch"]
            thickness = file.attrs["film_thickness"]
            alpha = file.attrs["alpha_bare"]
            data[thickness][pitch] = alpha
    
    sorted_data = dict(sorted(data.items()))

    fig, ax = plt.subplots(figsize=(12, 8))
    for thickness, inner_dict in sorted_data.items():
        pitches = np.array(list(inner_dict.keys())) * 1e6
        alphas = np.array(list(inner_dict.values()))
        ax.scatter(pitches, alphas, label=f"{round(thickness * 1e9)}nm")
    ax.set(xlabel=r"CPW gap ($\mu$m)", ylabel=r"KI fraction $\alpha$")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(bbox_to_anchor=(1.05, 1))
    fig.tight_layout()
    plt.show()
