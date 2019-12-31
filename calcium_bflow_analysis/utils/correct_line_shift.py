import pathlib

import matplotlib.pyplot as plt
import numpy as np
import imageio


def correct_line_shift(img: np.ndarray, value: int):
    """Corrects the lineshift of a given image."""
    rolled = np.roll(img[::2, :], value, axis=1)
    img[::2, :] = rolled
    return img


def show_corrected_image(img: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(corrected, cmap="gray")
    ax.axis("off")
    return fig, ax


if __name__ == "__main__":
    fname = pathlib.Path("/data/Amit_QNAP/Calcium_FXS/x10/")
    images = [
        next((fname / "WT_674").glob("AVG*WT*.png")),
        next((fname / "FXS_614").glob("AVG*FXS*.png")),
    ]
    for image in images:
        data = imageio.imread(image)
        corrected = correct_line_shift(data, 3)
        fig, ax = show_corrected_image(corrected)
        fig.savefig(image.with_suffix(".corrected.png"), transparent=True, dpi=300)
