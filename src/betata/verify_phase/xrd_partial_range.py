""" """

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from betata import plt, get_purples
import matplotlib.ticker as tck

TRACE_COLOR = get_purples(1, 1.0, 1.0)[0]
TRANSPARENCY = 0.85


@dataclass
class XRDScan:
    """ """

    path: Path
    angle: np.ndarray = None
    intensity: np.ndarray = None
    domain: tuple[float, float] = None


@dataclass
class RefPeak:
    """ """

    label: str
    location: float = None  # degrees
    label_xloc: float = None
    label_yloc: float = None


def extract_data(filepath):
    """ """
    header = "[Data]"
    header_idx = None

    with open(filepath) as file:
        for idx, line in enumerate(file.readlines(), start=1):
            if line.startswith(header):
                header_idx = idx
                break

    data = np.genfromtxt(
        filepath,
        skip_header=header_idx,
        usecols=(0, 1),
        delimiter=",",
        names=True,
    )

    data.dtype.names = "2θ", "Intensity"
    angle, intensity = data["2θ"], data["Intensity"]
    return angle, intensity


def plot_data(
    scan: XRDScan,
    ref_peaks: list[RefPeak] = None,
    figsize=(6, 6),
    yscale="log",
):
    """ """
    ref_peaks = [] if ref_peaks is None else ref_peaks

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_yscale(yscale)
    ax.set_xlabel(r"2$\mathrm{\theta}$ (°)")
    ax.set_ylabel("Intensity (A.U.)")

    if scan.domain is not None:
        left = np.argmin(np.abs(scan.angle - scan.domain[0]))
        right = np.argmin(np.abs(scan.angle - scan.domain[1]))
    else:
        left, right = 0, len(scan.angle) - 1

    ax.plot(
        scan.angle[left:right],
        scan.intensity[left:right],
        color=TRACE_COLOR,
        alpha=TRANSPARENCY,
    )

    for peak in ref_peaks:
        ax.annotate(
            f"{peak.label}",
            (peak.label_xloc, peak.label_yloc),
            rotation="vertical",
            verticalalignment="bottom",
            horizontalalignment="center",
        )

    ax.set_yticks([])
    ax.xaxis.set_major_locator(tck.MultipleLocator(2))
    ax.xaxis.set_minor_locator(tck.MultipleLocator(0.2))

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    """ """

    scan = XRDScan(
        path=Path(__file__).parents[3] / "data/verify_phase/XRD_066.dql",
        domain=[30, 46],
    )

    scan.angle, scan.intensity = extract_data(scan.path)

    ATA110 = RefPeak(label=r"   $\alpha$-Ta (110)")
    ATA110_domain_mask = (scan.angle > 38) & (scan.angle < 38.3)
    AT110_domain = scan.angle[ATA110_domain_mask]
    AT110_argloc = np.argmax(scan.intensity[ATA110_domain_mask])
    ATA110.location = AT110_domain[AT110_argloc]
    ATA110.label_xloc = ATA110.location
    ATA110.label_yloc = scan.intensity[ATA110_domain_mask][AT110_argloc]

    print(ATA110)

    BTA002 = RefPeak(label=r"   $\beta$-Ta (002)")
    BTA002_domain_mask = (scan.angle > 33) & (scan.angle < 34)
    BTA002_domain = scan.angle[BTA002_domain_mask]
    BTA002_argloc = np.argmax(scan.intensity[BTA002_domain_mask])
    BTA002.location = BTA002_domain[BTA002_argloc]
    BTA002.label_xloc = BTA002.location
    BTA002.label_yloc = scan.intensity[BTA002_domain_mask][BTA002_argloc]

    print(BTA002)

    AL2O3110 = RefPeak(label=r"   Al$_2$O$_3$ (0006)")
    AL203110_domain_mask = (scan.angle > 41) & (scan.angle < 42)
    AL203110_domain = scan.angle[AL203110_domain_mask]
    AL203110_argloc = np.argmax(scan.intensity[AL203110_domain_mask])
    AL2O3110.location = AL203110_domain[AL203110_argloc]
    AL2O3110.label_xloc = AL2O3110.location + 1
    AL2O3110.label_yloc = BTA002.label_yloc - 500

    print(AL2O3110)

    figure = plot_data(
        scan,
        ref_peaks=[ATA110, BTA002, AL2O3110],
        yscale="linear",
        figsize=(8, 5),
    )

    figsavepath = Path(__file__).parents[3] / "out/verify_phase/XRD.png"

    plt.savefig(figsavepath, dpi=300, bbox_inches="tight")

    plt.show()
