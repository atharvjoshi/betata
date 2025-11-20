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
    
    input_folder = Path(__file__).parents[3] / "out/resonator_studies"

    data = defaultdict(dict)

    for filepath in input_folder.iterdir():

        if filepath.suffix not in [".h5", ".hdf5"]:
            continue

        with h5py.File(filepath) as file:

            if file.attrs["type"] == "LE":
                continue

            pitch = file.attrs["pitch"]
            thickness = file.attrs["film_thickness"]
            fr_geom = file.attrs["fr_geom"]
            fr_meas = file[list(file.keys())[0]].attrs["fr"]
            alpha = get_alpha(fr_meas, fr_geom)

            data[thickness][pitch] = alpha

    fig, ax = plt.subplots(figsize=(8, 6))
    for thickness, inner_dict in data.items():
        pitches = np.array(list(inner_dict.keys())) * 1e6
        alphas = np.array(list(inner_dict.values()))
        ax.scatter(pitches, alphas, label=f"{round(thickness * 1e9)}nm")
    ax.set(xlabel="Pitch (um)", ylabel=r"$\alpha$")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
    ax.legend()
    fig.tight_layout()
    plt.show()
