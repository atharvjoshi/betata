"""

Find bare resonator frequency and calculate kinetic inductance fraction alpha and kinetic inductance

"""

import numpy as np
from uncertainties import ufloat

from betata.resonator_studies.resonator import (
    Resonator,
    load_resonators,
    save_resonator,
)

# assign a reasonable-ish uncertainty for the simulated frequency
SIM_UNCERTAINTY = 0.05
# since circle fit errors are negligible and do not capture the true uncertainty, assign a reasonable-ish uncertainty for measured frequency
FR_BARE_UNCERTAINTY = 1e-5


def find_alpha_bare(fr_geom: float, fr_bare: float) -> float:
    """ """
    return 1 - (fr_bare / fr_geom) ** 2


def find_l_kin(l_geom: float, alpha: float) -> float:
    """ """
    return (l_geom * alpha) / (1 - alpha)


def find_fr_bare(resonator: Resonator) -> float | None:
    """
    """
    MAX_TEMP = 15e-3
    MIN_POWER, MAX_POWER = -50, -40
    # traces are pre-sorted by power (decreasing) and temperature (increasing)
    frs_bare = []
    for trace in resonator.traces:
        if not trace.is_excluded:
            if trace.temperature <= MAX_TEMP and MIN_POWER <= trace.power <= MAX_POWER:
                frs_bare.append(trace.fr)
    return np.mean(np.array(frs_bare))


if __name__ == "__main__":
    """ """

    resonators: list[Resonator] = load_resonators()

    for resonator in resonators:
        fr_bare = find_fr_bare(resonator)
        u_fr_bare = ufloat(fr_bare, FR_BARE_UNCERTAINTY * fr_bare)
        u_fr_geom = ufloat(resonator.fr_geom, SIM_UNCERTAINTY * resonator.fr_geom)
        u_alpha_bare = find_alpha_bare(u_fr_geom, u_fr_bare)
        u_l_geom = ufloat(resonator.l_geom, SIM_UNCERTAINTY * resonator.l_geom)
        u_l_kin = find_l_kin(u_l_geom, u_alpha_bare)

        resonator.fr_bare = u_fr_bare.n
        resonator.alpha_bare = u_alpha_bare.n
        resonator.alpha_bare_err = u_alpha_bare.s
        resonator.l_kin = u_l_kin.n
        resonator.l_kin_err = u_l_kin.s

        print(resonator.name)
        print(u_l_kin * 1e9)

        save_resonator(resonator)
