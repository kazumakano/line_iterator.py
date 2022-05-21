from __future__ import annotations
from enum import IntEnum
from typing import Any, Literal
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import line

class ColorSpace(IntEnum):
    BGR = 0
    GRAY = 1
    RGB = 2

def _show_rgb_bar(ax: plt.Axes, line_img: np.ndarray) -> None:
    ax.set_yticks(())
    ax.imshow(line_img, aspect="auto")

def _show_gray_bar(ax: plt.Axes, line_img: np.ndarray) -> None:
    ax.set_yticks(())
    ax.imshow(line_img, cmap="gray", aspect="auto", vmin=0, vmax=255)

def _plot_rgb_values(axes: np.ndarray, line_iterator: LineIterator) -> None:
    colors = ("red", "green", "blue")
    for i, a in enumerate(reversed(axes)):
        a: plt.Axes
        a.set_ylim(0, 255)
        a.set_yticks((0, 255))
        a.plot(line_iterator[:, i], color=colors[i])

def _plot_gray_values(ax: plt.Axes, line_iterator: LineIterator) -> None:
    ax.set_ylim(0, 255)
    ax.set_yticks((0, 255))
    ax.plot(line_iterator, color="black")

class LineIterator(np.ndarray):
    def __new__(cls, img: np.ndarray, pt1: Any, pt2: Any) -> LineIterator:
        if not isinstance(img, np.ndarray):
            raise Exception(f"type of image is expected to be 'numpy.ndarray' but {type(img)} was given")
        if img.dtype != np.uint8:
            raise Exception(f"data type of image is expected to be 'uint8' but {img.dtype} was given")
        if img.ndim not in (2, 3):
            raise Exception(f"the number of dimensions of image is expected to be 2 or 3 but {img.ndim} was given")
        if len(pt1) != 2 or len(pt2) != 2:
            raise Exception(f"length of points are expected to be 2 but {len(pt1)} and {len(pt2)} were given")

        offsets: tuple[np.ndarray, np.ndarray] = line(pt1[1], pt1[0], pt2[1], pt2[0])
        self = np.array(img[offsets[0], offsets[1]])
        self.setflags(write=False)    # set immutable

        return self.view(type=cls)

    def show(self, color_space: Literal[ColorSpace.BGR, ColorSpace.GRAY, ColorSpace.RGB] = ColorSpace.BGR) -> None:
        match color_space:
            case ColorSpace.BGR:
                if self.ndim != 2:
                    raise Exception(f"the number of dimensions is not 2 but {self.ndim}")

                axes: np.ndarray = plt.subplots(nrows=4, sharex=True)[1]
                line_img: np.ndarray = cv2.cvtColor(self[np.newaxis], cv2.COLOR_BGR2RGB)    # convert from BGR to RGB
                _show_rgb_bar(axes[0], line_img)
                _plot_rgb_values(axes[1:], line_img[0])

            case ColorSpace.RGB:
                if self.ndim != 2:
                    raise Exception(f"the number of dimensions is not 2 but {self.ndim}")

                axes: np.ndarray = plt.subplots(nrows=4, sharex=True)[1]
                _show_rgb_bar(axes[0], self[np.newaxis])
                _plot_rgb_values(axes[1:], self)
            
            case ColorSpace.GRAY:
                if self.ndim != 1:
                    raise Exception(f"the number of dimensions is not 1 but {self.ndim}")

                axes: np.ndarray = plt.subplots(nrows=2, sharex=True)[1]
                _show_gray_bar(axes[0], self[np.newaxis])
                _plot_gray_values(axes[1], self)

            case _:
                raise Exception(f"unexpected color space {color_space} was given")

        plt.show()
        plt.close()
