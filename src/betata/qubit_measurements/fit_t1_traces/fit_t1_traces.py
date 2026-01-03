""" """

from pathlib import Path

import numpy as np
import lmfit

from betata import plt
from betata.qubit_measurements.qubit import Qubit
from betata.qubit_measurements.traces import T1Trace


def t1_fit_fn(x, A, T1, B):
    """ """
    return A * np.exp(-x / T1) + B


class T1Model(lmfit.Model):
    """ """

    def __init__(self, *args, **kwargs):
        """ """
        name = self.__class__.__name__
        super().__init__(func=t1_fit_fn, name=name, *args, **kwargs)

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
        A_guess = data[0] - data[-1]
        B_guess = data[-1]
        popn_T1_idx = np.abs(data - ((A_guess / np.e) + B_guess)).argmin()
        T1_guess = x[popn_T1_idx]

        guesses = {
            "A": {"value": A_guess, "min": 0, "max": 1},
            "T1": {"value": T1_guess, "min": 0},
            "B": {"value": B_guess, "min": 0, "max": 1},
        }
        return self.make_params(guesses=guesses)


def fit_t1_trace(
    trace: T1Trace,
    plot=True,
    verbose=False,
    save_folder=None,
    method="leastsq",
) -> lmfit.model.ModelResult:
    """ """
    tau = trace.tau  # seconds
    population = trace.population

    fit_result = T1Model().fit(population, tau, verbose=verbose, method=method)

    trace.T1 = fit_result.params["T1"].value
    trace.T1_err = fit_result.params["T1"].stderr
    trace.A = fit_result.params["A"].value
    trace.A_err = fit_result.params["A"].stderr
    trace.B = fit_result.params["B"].value
    trace.B_err = fit_result.params["B"].stderr

    if not plot:
        return fit_result

    fig, _ = plot_t1_trace(trace)

    if save_folder is not None:
        ts_str = trace.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"{ts_str}_{trace.id}_{trace.qubit_name}_T1.jpg"
        save_filepath = Path(save_folder) / save_filename
        plt.savefig(save_filepath, dpi=50, bbox_inches="tight")
        plt.close(fig)

    return fit_result


def plot_t1_trace(trace: T1Trace, show_fit=True, figsize=(5, 5)):
    """ """
    tau_us = trace.tau * 1e6

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(tau_us, trace.population, color="k", alpha=0.8, zorder=-1)

    T1_str = None
    if None not in [trace.T1, trace.T1_err, trace.A, trace.B] and show_fit:
        T1_us = trace.T1 * 1e6
        T1_err_us = trace.T1_err * 1e6
        T1_str = f"{T1_us:.2f} ± {T1_err_us:.2f} μs"
        T1_popn = (trace.A / np.e) + trace.B
        best_fit = t1_fit_fn(trace.tau, trace.A, trace.T1, trace.B)
        ax.plot(tau_us, best_fit, color="r")
        ax.axhline(y=T1_popn, color="g", zorder=-2, alpha=0.5)
        ax.axvline(x=T1_us, color="g", zorder=-2, alpha=0.5)

    ax.set_title(f"{trace.timestamp} #{trace.id} \n T1 = {T1_str}")
    ax.set_xlabel(r"$\tau$ (μs)")
    ax.set_ylabel("$P_e$")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.00])
    ax.set_xscale("log")

    fig.tight_layout()

    return fig, ax


def plot_t1_vs_time(traces: list[T1Trace], qubit_name: str):
    """ """

    t1_timestamp = np.array(
        [(np.abs(tr.timestamp - traces[0].timestamp)).total_seconds() for tr in traces]
    )
    t1_timestamp_hr = t1_timestamp / 3600

    t1 = np.array([tr.T1 for tr in traces])
    t1_err = np.array([tr.T1_err for tr in traces])
    t1_us, t1_err_us = t1 * 1e6, t1_err * 1e6

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(
        t1_timestamp_hr,
        t1_us,
        yerr=t1_err_us,
        ls="",
        color="k",
        marker="o",
        alpha=0.85,
        zorder=-1,
    )

    t1_avg, t1_avg_err = np.mean(t1), np.std(t1)
    t1_avg_us, t1_avg_err_us = t1_avg * 1e6, t1_avg_err * 1e6

    ax.axhline(t1_avg_us, ls="--", color="r")
    ax.axhline(t1_avg_us - t1_avg_err_us / 2, ls="--", color="r", alpha=0.5)
    ax.axhline(t1_avg_us + t1_avg_err_us / 2, ls="--", color="r", alpha=0.5)

    avg_t1_str = r"$\overline{T_1} = $"
    avg_t1_str += f"{t1_avg_us:.1f} ± {t1_avg_err_us:.1f} μs"
    ax.set_title(f"{qubit_name}: {avg_t1_str}")

    ax.set_xlabel("Time (hour)")
    ax.set_ylabel(r"$T_1$ (μs)")

    fig.tight_layout()

    return fig
