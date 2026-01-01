""" """

from pathlib import Path

from lmfit import Model
import numpy as np
from uncertainties import ufloat, umath
from scipy.constants import physical_constants

from betata import plt
from betata.qubit_measurements.traces import RPMTrace, load_rpm_trace

DATA_FOLDER = Path(__file__).parents[3] / "data/qubit_measurements"

COLOR_G, COLOR_E = "0.15", "0.55"
TRANSPARENCY = 0.85


def rpm_fit_fn(x, A, f, phi, B):
    """ """
    return A * np.cos(2 * np.pi * f * x + phi) + B


class RPMModel(Model):
    """ """

    def __init__(self, *args, **kwargs):
        """ """
        name = self.__class__.__name__
        super().__init__(func=rpm_fit_fn, name=name, *args, **kwargs)

    def fit(self, data, x, params=None, verbose=False, **kwargs):
        """ """
        if params is None:
            params = self.guess(data, x)
        result = super().fit(data, params=params, x=x, **kwargs)
        if verbose:
            print(result.fit_report())
        return result

    def make_params(self, guesses: dict = None, **kwargs):
        """ """
        if guesses is not None:
            for param, hint in guesses.items():
                self.set_param_hint(param, **hint)
        return super().make_params(**kwargs)

    def guess(self, data, x):
        """ """
        A_guess = np.abs(np.min(data) - np.max(data))

        data_fft = np.fft.fft(data - np.mean(data))
        data_freqs = np.fft.fftfreq(len(data), x[1] - x[0])
        f_guess = np.abs(data_freqs[np.argmax(np.abs(data_fft[1:])) + 1])

        phi_guess, B_guess = 0, 0

        guesses = {
            "A": {"value": A_guess, "min": 0},
            "f": {"value": f_guess, "min": 0},
            "phi": {"value": phi_guess, "min": -np.pi, "max": np.pi},
            "B": {"value": B_guess},
        }
        return self.make_params(guesses=guesses)


def signal_norm(signal, reference):
    """ """
    return 2 * (signal - np.mean(signal)) / (np.max(reference) - np.min(reference))


def calculate_qubit_temperature(a_g, a_g_err, a_e, a_e_err, freq_q):
    """ """
    u_a_g = ufloat(a_g, a_g_err)
    u_a_e = ufloat(a_e, a_e_err)

    p_g = u_a_g / (u_a_e + u_a_g)
    p_e = u_a_e / (u_a_e + u_a_g)
    popn_ratio = p_g / p_e

    kb_Hz_K, _, _ = physical_constants["Boltzmann constant in Hz/K"]

    qubit_temperature_K = freq_q / (kb_Hz_K * umath.log(popn_ratio))

    return qubit_temperature_K


if __name__ == "__main__":
    """ """

    filepath = DATA_FOLDER / "2025-12-02_11-27-42_Q6_4p69_rpm.h5"
    figsavepath = Path(__file__).parents[3] / "out/qubit_measurements/rpm.png"

    rpm_trace: RPMTrace = load_rpm_trace(filepath)

    amplitude = rpm_trace.amplitude
    mag_g = np.mean(np.sqrt(rpm_trace.I_g**2 + rpm_trace.Q_g**2), axis=0)
    mag_e = np.mean(np.sqrt(rpm_trace.I_e**2 + rpm_trace.Q_e**2), axis=0)

    fit_model = RPMModel()
    fit_result_g = fit_model.fit(mag_g, amplitude, verbose=True)
    fit_result_e = fit_model.fit(mag_e, amplitude, verbose=True)
    best_fit_g, best_fit_e = fit_result_g.best_fit, fit_result_e.best_fit

    a_g = fit_result_g.params["A"].value
    a_g_err = fit_result_g.params["A"].stderr
    a_e = fit_result_e.params["A"].value
    a_e_err = fit_result_e.params["A"].stderr
    freq_q = rpm_trace.qubit_frequency
    qubit_temperature = calculate_qubit_temperature(a_g, a_g_err, a_e, a_e_err, freq_q)
    qubit_temperature_mK = qubit_temperature * 1e3
    print(f"Qubit temperature: {qubit_temperature_mK:.2f} mK")

    mag_g_norm = signal_norm(mag_g, mag_g)
    mag_e_norm = signal_norm(mag_e, mag_g)
    best_fit_g_norm = signal_norm(best_fit_g, best_fit_g)
    best_fit_e_norm = signal_norm(best_fit_e, best_fit_g)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel(r"$|e \rangle \to |f \rangle$ drive amplitude (A.U.)")
    ax.set_ylabel("Normalized signal amplitude (A.U.)")

    ax.scatter(amplitude, mag_g_norm, label=r"$A_g$", c=COLOR_G, alpha=TRANSPARENCY)
    ax.plot(amplitude, best_fit_g_norm, color=COLOR_G)

    ax.scatter(
        amplitude,
        mag_e_norm,
        label=r"$A_e$",
        c=COLOR_E,
        alpha=TRANSPARENCY,
        zorder=-1,
    )
    ax.plot(amplitude, best_fit_e_norm, color=COLOR_E, zorder=-1)

    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

    fig.tight_layout()

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
