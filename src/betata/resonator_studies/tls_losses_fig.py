""" """

from betata import plt
from pathlib import Path
from PIL import Image

if __name__ == "__main__":
    """ """

    fig_folder = Path(__file__).parents[3] / "out/resonator_studies"
    fig_3a_path = fig_folder / "circle_fit.png"
    fig_3b_path = fig_folder / "power_temp_sweep.png"
    fig_3c_path = fig_folder / "wang_plot.png"

    image_3a = Image.open(fig_3a_path)
    image_3b = Image.open(fig_3b_path)
    image_3c = Image.open(fig_3c_path)

    print(image_3a.size)
    print(image_3b.size)
    print(image_3c.size)

    images = [image_3a, image_3b, image_3c]
    labels = ["(a)", "(b)", "(c)"]

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[image_3a.width, image_3b.width],
        height_ratios=[image_3a.height, image_3c.height],
    )
    ax_circle_fit = fig.add_subplot(gs[0, 0])
    ax_qpt_sweep = fig.add_subplot(gs[0, 1])
    ax_qtls0_pms = fig.add_subplot(gs[1, :])

    axes = [ax_circle_fit, ax_qpt_sweep, ax_qtls0_pms]

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")

        ax.text(
            -0.02,
            0.98,
            labels[i],
            transform=ax.transAxes,
            size=20,
            va="top",
            ha="left",
        )

    figsavepath = Path(__file__).parents[3] / "out/resonator_studies/fig3.png"
    plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
