""" """

from pathlib import Path

import numpy as np
import lmfit
from scipy.signal import find_peaks

from betata import plt
from betata.qubit_measurements.traces import T2RTrace


def t2r_fit_fn(x, **params: lmfit.Parameter):
    """ """
    T2R = params["T2R"]
    B = params["B"]
    As = np.array([v for k, v in params.items() if k.startswith("A")])
    freqs = np.array([v for k, v in params.items() if k.startswith("f")])
    sum_cos = sum(As[i] * np.cos(2 * np.pi * freqs[i] * x) for i in range(len(freqs)))
    return np.exp(-x / T2R) * sum_cos + B


class T2RModel(lmfit.Model):
    """ """

    def __init__(self, *args, **kwargs):
        """ """
        name = self.__class__.__name__
        super().__init__(func=t2r_fit_fn, name=name, *args, **kwargs)

    def fit(self, data, x, params, verbose=False, **kwargs):
        """ """
        result = super().fit(data, params=params, x=x, **kwargs)
        if verbose:
            print(result.fit_report())
        return result


def fit_t2r_trace(
    trace: T2RTrace,
    max_n_freqs: int = 4,
    freq_peak_threshold: float = 0.3,
    plot=True,
    verbose=False,
    save_folder=None,
    method="leastsq",
    params=None,
) -> lmfit.model.ModelResult:
    """ """
    tau = trace.tau  # seconds
    population = trace.population

    # FFT
    N = len(tau)
    dt = tau[1] - tau[0]
    yf = np.abs(np.fft.fft(population - np.mean(population)))
    xf = np.fft.fftfreq(N, dt)

    xf_pos = xf[xf > 0]
    yf_pos = yf[xf > 0]

    peaks, _ = find_peaks(yf_pos)
    peak_amps = yf_pos[peaks]
    if len(peak_amps) == 0:
        print(f"[Skipped] T2RTrace#{trace.id}: found no FFT peaks")
        return

    main_amp = peak_amps.max()
    good_mask = peak_amps >= freq_peak_threshold * main_amp
    good_peaks = peaks[good_mask]
    good_amps = peak_amps[good_mask]

    n_freqs = min(len(good_peaks), max_n_freqs)
    freqs_init = xf_pos[good_peaks[:n_freqs]]
    amps_init = good_amps[:n_freqs]
    amps_norm = amps_init / np.sum(amps_init)

    if params is None:
        params = lmfit.Parameters()
        params.add("T2R", value=50e-6, min=1e-9)
        params.add("B", value=0.5, min=-1, max=1)
        for i in range(n_freqs):
            params.add(f"A{i}", value=amps_norm[i], min=-1, max=1)
            params.add(f"f{i}", value=freqs_init[i], min=0, max=1e6)

    fit_result = T2RModel().fit(
        population,
        tau,
        params=params,
        verbose=verbose,
        method=method,
    )

    fit_params = fit_result.params
    trace.T2R = fit_params["T2R"].value
    trace.T2R_err = fit_params["T2R"].stderr
    trace.As = [v.value for k, v in fit_params.items() if k.startswith("A")]
    trace.A_errs = [v.stderr for k, v in fit_params.items() if k.startswith("A")]
    trace.freqs = [v.value for k, v in fit_params.items() if k.startswith("f")]
    trace.freq_errs = [v.stderr for k, v in fit_params.items() if k.startswith("f")]
    trace.B = fit_result.params["B"].value
    trace.B_err = fit_result.params["B"].stderr

    if not plot:
        return fit_result

    fft_params = {"xf": xf_pos, "yf": yf_pos, "nf": n_freqs}

    fig, _, _ = plot_t2r_trace(trace, fit_params, fft_params)

    if save_folder is not None:
        ts_str = trace.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"{ts_str}_{trace.id}_{trace.qubit_name}_T2R.jpg"
        save_filepath = Path(save_folder) / save_filename
        plt.savefig(save_filepath, dpi=50, bbox_inches="tight")
        plt.close(fig)

    return fit_result


def plot_t2r_trace(trace: T2RTrace, fit_params=None, fft_params=None, figsize=(10, 7)):
    """ """
    tau_us = trace.tau * 1e6
    tau_dummy = np.linspace(min(trace.tau), max(trace.tau), 1001)
    tau_us_dummy = tau_dummy * 1e6

    ax1, ax2 = None, None
    if fft_params is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)

    ax1.scatter(tau_us, trace.population, color="k", alpha=0.8, zorder=-1)

    T2R_str = None
    if None not in [fit_params, trace.T2R, trace.T2R_err, trace.B]:
        T2R_us = trace.T2R * 1e6
        T2R_err_us = trace.T2R_err * 1e6
        T2R_str = f"{T2R_us:.2f} ± {T2R_err_us:.2f} μs"
        best_fit = t2r_fit_fn(tau_dummy, **fit_params)
        ax1.plot(tau_us_dummy, best_fit, color="r")

    fig_title = f"{trace.timestamp} #{trace.id} \n T2R = {T2R_str}"

    if fft_params is not None:
        ax2.plot(fft_params["xf"] * 1e-6, fft_params["yf"], c="k", alpha=0.8)
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Amplitude (A.U.)")
        fig_title += f" [F = {fft_params['nf']}]"

    ax1.set_xlabel(r"$\tau$ (μs)")
    ax1.set_ylabel("$P_e$")
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.suptitle(fig_title)

    return fig, ax1, ax2


def plot_t2r_vs_time(traces: list[T2RTrace], qubit_name: str):
    """ """

    t2r_timestamp = np.array(
        [(np.abs(tr.timestamp - traces[0].timestamp)).total_seconds() for tr in traces]
    )
    t2r_timestamp_hr = t2r_timestamp / 3600

    t2r = np.array([tr.T2R for tr in traces])
    t2r_err = np.array([tr.T2R_err for tr in traces])
    t2r_us, t2r_err_us = t2r * 1e6, t2r_err * 1e6

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(
        t2r_timestamp_hr,
        t2r_us,
        yerr=t2r_err_us,
        ls="",
        color="k",
        marker="o",
        alpha=0.85,
        zorder=-1,
    )

    t2r_avg, t2r_avg_err = np.mean(t2r), np.std(t2r)
    t2r_avg_us, t2r_avg_err_us = t2r_avg * 1e6, t2r_avg_err * 1e6

    ax.axhline(t2r_avg_us, ls="--", color="r")
    ax.axhline(t2r_avg_us - t2r_avg_err_us / 2, ls="--", color="r", alpha=0.5)
    ax.axhline(t2r_avg_us + t2r_avg_err_us / 2, ls="--", color="r", alpha=0.5)

    avg_t2r_str = r"$\overline{T_{2,R}} = $"
    avg_t2r_str += f"{t2r_avg_us:.1f} ± {t2r_avg_err_us:.1f} μs"
    ax.set_title(f"{qubit_name}: {avg_t2r_str}")

    ax.set_xlabel("Time (hour)")
    ax.set_ylabel(r"$T_{2,R}$ (μs)")

    fig.tight_layout()

    return fig
