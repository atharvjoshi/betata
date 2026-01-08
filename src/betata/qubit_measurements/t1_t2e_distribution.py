""" """

from pathlib import Path

import numpy as np
from matplotlib import ticker


from betata import plt
from betata.qubit_measurements.qubit import load_qubits

T1_TRACE_COLOR = "#E77500"
T2E_TRACE_COLOR = "#003D7C"

TRANSPARENCY = 0.85

if __name__ == "__main__":
    """ """

    figsavepath = (
        Path(__file__).parents[3] / "out/qubit_measurements/t1_t2e_distribution.png"
    )

    qubits = load_qubits()
    qubits = sorted(qubits, key=lambda qubit: qubit.f_q)

    all_t1 = [qubit.t1 for qubit in qubits]
    all_t1_us = [t1 * 1e6 for t1 in all_t1]
    all_omega_t1 = [(2 * np.pi * qubit.f_q * qubit.t1) for qubit in qubits]
    all_omega_t1_million = [omega_t1 * 1e-6 for omega_t1 in all_omega_t1]

    all_t2e = [qubit.t2e for qubit in qubits]
    all_t2e_us = [t2e * 1e6 for t2e in all_t2e]
    all_omega_t2e = [(2 * np.pi * qubit.f_q * qubit.t2e) for qubit in qubits]
    all_omega_t2e_million = [omega_t2e * 1e-6 for omega_t2e in all_omega_t2e]

    np.random.seed(11)

    fig, ax = plt.subplots(figsize=(18, 6))

    t1_vplot_parts = ax.violinplot(
        all_omega_t1_million,
        side="low",
        showextrema=False,
        showmeans=True,
        widths=1.0,
    )

    for part in t1_vplot_parts["bodies"]:
        part.set_facecolor(T1_TRACE_COLOR)
        part.set_edgecolor(T1_TRACE_COLOR)
        part.set_alpha(TRANSPARENCY)

    t1_vplot_parts["cmeans"].set_color(T1_TRACE_COLOR)

    t2e_vplot_parts = ax.violinplot(
        all_omega_t2e_million,
        side="high",
        showextrema=False,
        showmeans=True,
        widths=1.0,
    )

    for part in t2e_vplot_parts["bodies"]:
        part.set_facecolor(T2E_TRACE_COLOR)
        part.set_edgecolor(T2E_TRACE_COLOR)
        part.set_alpha(TRANSPARENCY)

    t2e_vplot_parts["cmeans"].set_color(T2E_TRACE_COLOR)

    ax.set_xlim(0.1, 11.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Qubit")

    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylabel(r"$\mathrm{\omega_q \times \{ T_1, T_{2, E} \} \; (\times 10^6)}$")

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")

    plt.show()
