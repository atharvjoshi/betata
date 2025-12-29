""" """

from betata import plt
from pathlib import Path
from PIL import Image

if __name__ == "__main__":
    """ """

    fig_folder = Path(__file__).parents[3] / "out/resonator_studies"
    fig_2ab_path = fig_folder / "alpha_pitch_fr_thickness.png"
    fig_2c_path = fig_folder / "penetration_depth.png"

    image_2ab = Image.open(fig_2ab_path)
    image_2c = Image.open(fig_2c_path)
    image_2c = image_2c.resize((image_2ab.width, image_2c.height))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        height_ratios=(image_2ab.height, image_2c.height),
        figsize=(8, 8),
    )

    # place images
    images = [image_2ab, image_2c]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis("off")

    label_fontsize = 16

    # place text
    axes[0].text(
        -0.06,
        0.97,
        "(a)",
        transform=axes[0].transAxes,
        size=label_fontsize,
        va="top",
        ha="left",
    )

    axes[0].text(
        0.44,
        0.97,
        "(b)",
        transform=axes[0].transAxes,
        size=label_fontsize,
        va="top",
        ha="left",
    )

    axes[1].text(
        -0.06,
        0.99,
        "(c)",
        transform=axes[1].transAxes,
        size=label_fontsize,
        va="top",
        ha="left",
    )

    fig.tight_layout()
    figsavepath = Path(__file__).parents[3] / "out/resonator_studies/fig2.png"
    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
