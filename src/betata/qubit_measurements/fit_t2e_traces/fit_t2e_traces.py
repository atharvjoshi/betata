""" """

from pathlib import Path

import numpy as np
import lmfit

from betata import plt
from betata.qubit_measurements.traces import T2ETrace


def t2e_fit_fn(x, A, T2E, B):
    """ """
    return A * (1 - np.exp(-x / T2E)) + B


class T2EModel(lmfit.Model):
    """ """

    def __init__(self, *args, **kwargs):
        """ """
        name = self.__class__.__name__
        super().__init__(func=t2e_fit_fn, name=name, *args, **kwargs)

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
        A_guess = data[-1] - data[0]
        B_guess = data[0]
        popn_T2E_idx = np.abs(data - ((A_guess * (1 - 1 / np.e)) + B_guess)).argmin()
        T2E_guess = x[popn_T2E_idx]

        guesses = {
            "A": {"value": A_guess, "min": 0, "max": 0.5},
            "T2E": {"value": T2E_guess, "min": 0},
            "B": {"value": B_guess, "min": 0, "max": 0.5},
        }
        return self.make_params(guesses=guesses)


def fit_t2e_trace(
    trace: T2ETrace,
    plot=True,
    verbose=False,
    save_folder=None,
    method="leastsq",
    params=None,
) -> lmfit.model.ModelResult:
    """ """
    tau = trace.tau  # seconds
    population = trace.population

    fit_result = T2EModel().fit(
        population,
        tau,
        verbose=verbose,
        method=method,
        params=params,
    )

    trace.T2E = fit_result.params["T2E"].value
    trace.T2E_err = fit_result.params["T2E"].stderr
    trace.A = fit_result.params["A"].value
    trace.A_err = fit_result.params["A"].stderr
    trace.B = fit_result.params["B"].value
    trace.B_err = fit_result.params["B"].stderr

    if not plot:
        return fit_result

    fig, _ = plot_t2e_trace(trace)

    if save_folder is not None:
        ts_str = trace.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"{ts_str}_{trace.id}_{trace.qubit_name}_T2E.jpg"
        save_filepath = Path(save_folder) / save_filename
        plt.savefig(save_filepath, dpi=50, bbox_inches="tight")
        plt.close(fig)

    return fit_result


def plot_t2e_trace(trace: T2ETrace, show_fit=True, figsize=(5, 5)):
    """ """
    tau_us = trace.tau * 1e6
    tau_us_dummy = np.linspace(min(tau_us), max(tau_us), 1001)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(tau_us, trace.population, color="k", alpha=0.8, zorder=-1)

    T2E_str = None
    if None not in [trace.T2E, trace.T2E_err, trace.A, trace.B] and show_fit:
        T2E_us = trace.T2E * 1e6
        T2E_err_us = trace.T2E_err * 1e6
        T2E_str = f"{T2E_us:.2f} ± {T2E_err_us:.2f} μs"
        T2E_popn = (trace.A * (1 - 1 / np.e)) + trace.B
        best_fit = t2e_fit_fn(tau_us_dummy, trace.A, T2E_us, trace.B)
        ax.plot(tau_us_dummy, best_fit, color="r")
        ax.axhline(y=T2E_popn, color="g", zorder=-2, alpha=0.5)
        ax.axvline(x=T2E_us, color="g", zorder=-2, alpha=0.5)

    ax.set_title(f"{trace.timestamp} #{trace.id} \n T2E = {T2E_str}")
    ax.set_xlabel(r"$\tau$ (μs)")
    ax.set_ylabel("$P_e$")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xscale("log")

    fig.tight_layout()

    return fig, ax


def plot_t2e_vs_time(traces: list[T2ETrace], qubit_name: str):
    """ """

    t2e_timestamp = np.array(
        [(np.abs(tr.timestamp - traces[0].timestamp)).total_seconds() for tr in traces]
    )
    t2e_timestamp_hr = t2e_timestamp / 3600

    t2e = np.array([tr.T2E for tr in traces])
    t2e_err = np.array([tr.T2E_err for tr in traces])
    t2e_us, t2e_err_us = t2e * 1e6, t2e_err * 1e6

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(
        t2e_timestamp_hr,
        t2e_us,
        yerr=t2e_err_us,
        ls="",
        color="k",
        marker="o",
        alpha=0.85,
        zorder=-1,
    )

    t2e_avg, t2e_avg_err = np.mean(t2e), np.std(t2e)
    t2e_avg_us, t2e_avg_err_us = t2e_avg * 1e6, t2e_avg_err * 1e6

    ax.axhline(t2e_avg_us, ls="--", color="r")
    ax.axhline(t2e_avg_us - t2e_avg_err_us / 2, ls="--", color="r", alpha=0.5)
    ax.axhline(t2e_avg_us + t2e_avg_err_us / 2, ls="--", color="r", alpha=0.5)

    avg_t2e_str = r"$\overline{T_{2,E}} = $"
    avg_t2e_str += f"{t2e_avg_us:.1f} ± {t2e_avg_err_us:.1f} μs"
    ax.set_title(f"{qubit_name}: {avg_t2e_str}")

    ax.set_xlabel("Time (hour)")
    ax.set_ylabel(r"$T_{2,E}$ (μs)")

    fig.tight_layout()

    return fig
