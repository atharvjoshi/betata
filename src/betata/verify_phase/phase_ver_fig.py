""" """

from betata import plt
from pathlib import Path
from PIL import Image

if __name__ == "__main__":
    """ """

    fig_folder = Path(__file__).parents[3] / "out/verify_phase"
    fig_1a_path = fig_folder / "TEM_HAADF.png"
    fig_1b_path = fig_folder / "XRD.png"
    fig_1c_path = fig_folder / "PPMS.png"

    image_1a = Image.open(fig_1a_path)
    image_1b = Image.open(fig_1b_path)
    image_1c = Image.open(fig_1c_path)

    # choose three-column (1, 3) figure layout
    # rescale 1a to match height of 1b and 1c
    image_1a_dim = image_1b.height
    image_1a = image_1a.resize((image_1a_dim, image_1a_dim))

    images = [image_1a, image_1b, image_1c]
    labels = ["(a)", "(b)", "(c)"]

    print(image_1a.height)
    print(image_1a.width)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        width_ratios=(image_1a.width, image_1b.width, image_1c.width),
        figsize=(13, 8),
    )

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis("off")

        ax.text(
            -0.11,
            1,
            labels[i],
            transform=ax.transAxes,
            size=10,
            va="top",
            ha="left",
        )

    figsavepath = Path(__file__).parents[3] / "out/verify_phase/fig1.png"

    # using tight layout adds unwanted whitespace
    #plt.savefig(figsavepath, dpi=600, bbox_inches="tight")

    plt.show()
