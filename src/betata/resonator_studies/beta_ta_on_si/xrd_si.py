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


@dataclass
class RefWavelength:
    """ """

    wavelength: float  # Angstrom
    label: str = None


def get_peak_location(ref_location, wavelength, primary_wavelength):
    """ """
    wl_ratio = wavelength / primary_wavelength
    return np.rad2deg(np.arcsin(wl_ratio * np.sin(np.deg2rad(ref_location / 2)))) * 2


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
    ref_wavelengths: list[RefWavelength] = None,
    figsize=(6, 6),
    yscale="log",
):
    """ """
    ref_peaks = [] if ref_peaks is None else ref_peaks
    pri_wl = ref_wavelengths[0]

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

    y_annotate = max(scan.intensity[left:right]) * 1.6

    for peak in ref_peaks:
        for wl in ref_wavelengths:
            wl_loc = get_peak_location(peak.location, wl.wavelength, pri_wl.wavelength)

            line_col = "k" if wl == pri_wl else "r"
            ax.axvline(
                wl_loc,
                c=line_col,
                lw=1,
                ls="--",
                alpha=0.75,
                zorder=-1,
                ymax=0.96,
            )

        ax.annotate(
            f"{peak.label}",
            (peak.label_xloc, y_annotate),
            rotation="vertical",
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=14,
        )

    # ax.set_yticks([])
    ax.yaxis.minorticks_off()
    ax.xaxis.set_major_locator(tck.MultipleLocator(10))
    ax.xaxis.set_minor_locator(tck.MultipleLocator(1))

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    """ """

    scan = XRDScan(
        path=Path(__file__).parents[4] / "data/beta_ta_on_si/XRD_211.dql",
        domain=[30, 80],
    )

    scan.angle, scan.intensity = extract_data(scan.path)

    ATA110 = RefPeak(label=r"$\alpha$-Ta (110)")
    ATA110_domain_mask = (scan.angle > 38) & (scan.angle < 38.3)
    AT110_domain = scan.angle[ATA110_domain_mask]
    AT110_argloc = np.argmax(scan.intensity[ATA110_domain_mask])
    ATA110.location = AT110_domain[AT110_argloc]
    ATA110.label_xloc = ATA110.location
    ATA110.label_yloc = scan.intensity[ATA110_domain_mask][AT110_argloc]

    print(ATA110.location)

    BTA002 = RefPeak(label=r"$\beta$-Ta (002)")
    BTA002_domain_mask = (scan.angle > 33) & (scan.angle < 34)
    BTA002_domain = scan.angle[BTA002_domain_mask]
    BTA002_argloc = np.argmax(scan.intensity[BTA002_domain_mask])
    BTA002.location = BTA002_domain[BTA002_argloc]
    BTA002.label_xloc = BTA002.location
    BTA002.label_yloc = scan.intensity[BTA002_domain_mask][BTA002_argloc]

    print(BTA002.location)

    BTA004 = RefPeak(label=r"$\beta$-Ta (004)")
    BTA004_domain_mask = (scan.angle > 70) & (scan.angle < 71)
    BTA004_domain = scan.angle[BTA004_domain_mask]
    BTA004_argloc = np.argmax(scan.intensity[BTA004_domain_mask])
    BTA004.location = BTA004_domain[BTA002_argloc]
    BTA004.label_xloc = BTA004.location + 1
    BTA004.label_yloc = scan.intensity[BTA004_domain_mask][BTA004_argloc]

    print(BTA004.location)

    SI400 = RefPeak(label=r"Si (400)")
    SI400_domain_mask = (scan.angle > 69) & (scan.angle < 70)
    SI400_domain = scan.angle[SI400_domain_mask]
    SI400_argloc = np.argmax(scan.intensity[SI400_domain_mask])
    SI400.location = SI400_domain[SI400_argloc]
    SI400.label_xloc = SI400.location
    SI400.label_yloc = scan.intensity[SI400_domain_mask][SI400_argloc]

    print(SI400.location)

    KA1 = RefWavelength(wavelength=1.54059292, label=r"K$\alpha_1$")
    KB = RefWavelength(wavelength=1.392246, label=r"K$\beta$")
    WLA1 = RefWavelength(wavelength=1.47631133, label=r"W-L$\alpha_1$")
    WLA2 = RefWavelength(wavelength=1.48745220, label=r"W-L$\alpha_2$")

    figure = plot_data(
        scan,
        ref_peaks=[BTA002, ATA110, SI400, BTA004],
        ref_wavelengths=[KA1, KB, WLA1],  # first entry is primary wavelength
        yscale="log",
        figsize=(8, 5),
    )

    figsavepath = Path(__file__).parents[4] / "out/beta_ta_on_si/XRD.svg"

    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
