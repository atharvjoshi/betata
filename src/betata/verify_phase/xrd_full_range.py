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
        path=Path(__file__).parents[3] / "data/verify_phase/XRD_066.dql",
        domain=[20, 110],
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

    ATA220 = RefPeak(label=r"$\alpha$-Ta (220)")
    ATA220_domain_mask = (scan.angle > 82) & (scan.angle < 83)
    AT220_domain = scan.angle[ATA220_domain_mask]
    AT220_argloc = np.argmax(scan.intensity[ATA220_domain_mask])
    ATA220.location = AT220_domain[AT220_argloc]
    ATA220.label_xloc = ATA220.location
    ATA220.label_yloc = scan.intensity[ATA220_domain_mask][AT220_argloc]

    print(ATA220.location)

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
    BTA004.label_xloc = BTA004.location
    BTA004.label_yloc = scan.intensity[BTA004_domain_mask][BTA004_argloc]

    print(BTA004.location)

    AL2O3110 = RefPeak(label=r"Al$_2$O$_3$ (0006)")
    AL203110_domain_mask = (scan.angle > 41) & (scan.angle < 42)
    AL203110_domain = scan.angle[AL203110_domain_mask]
    AL203110_argloc = np.argmax(scan.intensity[AL203110_domain_mask])
    AL2O3110.location = AL203110_domain[AL203110_argloc]
    AL2O3110.label_xloc = AL2O3110.location
    AL2O3110.label_yloc = scan.intensity[AL203110_domain_mask][AL203110_argloc]

    print(AL2O3110.location)

    AL2O3220 = RefPeak(label=r"Al$_2$O$_3$ (00012)")
    AL203220_domain_mask = (scan.angle > 90) & (scan.angle < 91)
    AL203220_domain = scan.angle[AL203220_domain_mask]
    AL203220_argloc = np.argmax(scan.intensity[AL203220_domain_mask])
    AL2O3220.location = AL203220_domain[AL203220_argloc]
    AL2O3220.label_xloc = AL2O3220.location
    AL2O3220.label_yloc = scan.intensity[AL203220_domain_mask][AL203220_argloc]

    print(AL2O3220.location)

    KA1 = RefWavelength(wavelength=1.54059292, label=r"K$\alpha_1$")
    KB = RefWavelength(wavelength=1.392246, label=r"K$\beta$")
    WLA1 = RefWavelength(wavelength=1.47631133, label=r"W-L$\alpha_1$")
    WLA2 = RefWavelength(wavelength=1.48745220, label=r"W-L$\alpha_2$")

    figure = plot_data(
        scan,
        ref_peaks=[BTA002, ATA110, AL2O3110, BTA004, ATA220, AL2O3220],
        ref_wavelengths=[KA1, KB, WLA1],  # first entry is primary wavelength
        yscale="log",
        figsize=(12, 6),
    )

    figsavepath = Path(__file__).parents[3] / "out/verify_phase/XRD_full_range.png"

    #plt.savefig(figsavepath, dpi=300, bbox_inches="tight")

    plt.show()
