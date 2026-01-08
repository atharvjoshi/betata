""" """

from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np

from betata import plt, get_purples
from betata.qubit_measurements.qubit import load_qubits, Qubit

TRACE_COLOR = get_purples(1, 1.0, 1.0)[0]

QUBITS_TO_INCLUDE = {
    "Q1_2p61",
    "Q2_2p80",
    "Q3_2p88",
    "Q4_3p02",
    "Q5_3p19",
    "Q6_4p69",
    "Q8_4p93",
    "Q9_5p15",
    "Q7_4p83",
    "Q10_5p36",
    "Q11_5p78",
}

# for these qubits, only include t1 and t2e obtained in the same measurement run
# qubit_name : [(t1_start_id, t1_stop_id), (t2e_start_id, t2e_stop_id)]
SAMPLES_TO_INCLUDE = {
    "Q1_2p61": [(511, 710), (336, 551)],
    "Q3_2p88": [(337, 456), (206, 343)],
    "Q4_3p02": [(614, 835), (339, 558)],
    "Q5_3p19": [(568, 772), (330, 533)],
}

if __name__ == "__main__":
    """ """

    figsavepath = Path(__file__).parents[3] / "out/qubit_measurements/t1_vs_t2e.png"

    all_qubits = load_qubits()
    included_qubits: list[Qubit] = []
    for qubit in all_qubits:
        if qubit.name in QUBITS_TO_INCLUDE:
            included_qubits.append(qubit)

    """
    for qubit in included_qubits:
        if qubit.name in SAMPLES_TO_INCLUDE.keys():
            t1_start_id, t1_stop_id = SAMPLES_TO_INCLUDE[qubit.name][0]
            t1_to_include = qubit.t1[t1_start_id:t1_stop_id]
            qubit.t1_avg = np.mean(t1_to_include)
            qubit.t1_avg_err = np.std(t1_to_include)

            t2e_start_id, t2e_stop_id = SAMPLES_TO_INCLUDE[qubit.name][1]
            t2e_to_include = qubit.t2e[t2e_start_id:t2e_stop_id]
            qubit.t2e_avg = np.mean(t2e_to_include)
            qubit.t2e_avg_err = np.std(t2e_to_include)
    """

    omega_q = np.array([2 * np.pi * qubit.f_q for qubit in included_qubits])

    omega_t1 = omega_q * np.array([qubit.t1_avg for qubit in included_qubits])
    omega_t1_err = omega_q * np.array([qubit.t1_avg_err for qubit in included_qubits])

    omega_t1_million, omega_t1_err_million = omega_t1 * 1e-6, omega_t1_err * 1e-6

    omega_t2e = omega_q * np.array([qubit.t2e_avg for qubit in included_qubits])
    omega_t2e_err = omega_q * np.array([qubit.t2e_avg_err for qubit in included_qubits])

    omega_t2e_million, omega_t2e_err_million = omega_t2e * 1e-6, omega_t2e_err * 1e-6

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.errorbar(
        x=omega_t1_million,
        y=omega_t2e_million,
        xerr=omega_t1_err_million,
        yerr=omega_t2e_err_million,
        marker="o",
        ls="",
        color=TRACE_COLOR,
    )

    ax.set_xlabel(r"$\mathrm{\omega_q \times \overline{T}_1 \; (\times 10^6)}$")
    ax.set_ylabel(r"$\mathrm{\omega_q \times \overline{T}_{2, E} \; (\times 10^6)}$")

    ax.set_xlim(0, 12)
    ax.set_xticks([0, 5, 10])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    ax.set_ylim(0, 12)
    ax.set_yticks([0, 5, 10])
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))

    q_dummy = np.linspace(0, 12, 51)
    ax.plot(q_dummy, q_dummy, color="0.5", ls="--")  # t2e = t1
    ax.plot(q_dummy, 2 * q_dummy, color="0.5", ls="--")  # t2e = 2 * t1

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")

    plt.show()
