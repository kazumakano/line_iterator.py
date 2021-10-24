from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import line

BGR: int = 0
RGB: int = 1
GRAY: int = 2
VMIN = np.uint8(0)
VMAX = np.uint8(255)

class LineIterator(np.ndarray):
    def __new__(cls: type, img: np.ndarray, pt1: Any, pt2: Any) -> LineIterator:
        if img.dtype != np.uint8:
            raise Exception("data type of image is not 'uint8'")

        offsets: tuple[np.ndarray, np.ndarray] = line(pt1[0], pt1[1], pt2[0], pt2[1])

        return np.array(img[offsets[0], offsets[1]]).view(type=cls)

    def show(self, color_space: int = BGR) -> None:
        if color_space == BGR:
            axes: np.ndarray = plt.subplots(nrows=4, sharex=True)[1]
            line_img: np.ndarray = cv2.cvtColor(self[np.newaxis], cv2.COLOR_BGR2RGB)    # convert from BGR to RGB
            _show_rgb_bar(axes[0], line_img)
            _plot_rgb_value(axes[1:], line_img[0])

        elif color_space == RGB:
            axes: np.ndarray = plt.subplots(nrows=4, sharex=True)[1]
            _show_rgb_bar(axes[0], self[np.newaxis])
            _plot_rgb_value(axes[1:], self)
        
        elif color_space == GRAY:
            axes: np.ndarray = plt.subplots(nrows=2, sharex=True)[1]
            _show_gray_bar(axes[0], self[np.newaxis])
            _plot_gray_value(axes[1], self)

        plt.show()

def _show_rgb_bar(ax: plt.Axes, line_img: np.ndarray) -> None:
    ax.set_yticks(())
    ax.imshow(line_img, aspect="auto")

def _show_gray_bar(ax: plt.Axes, line_img: np.ndarray) -> None:
    ax.set_yticks(())
    ax.imshow(line_img, cmap="gray", aspect="auto", vmin=VMIN, vmax=VMAX)

def _plot_rgb_value(axes: np.ndarray, line_iterator: LineIterator) -> None:
    rgb = ("red", "green", "blue")
    for i, a in enumerate(reversed(axes)):
        a.set_ylim(VMIN, VMAX)
        a.set_yticks((VMIN, VMAX))
        a.plot(line_iterator[:, i], color=rgb[i])

def _plot_gray_value(ax: plt.Axes, line_iterator: LineIterator) -> None:
    ax.set_ylim(VMIN, VMAX)
    ax.set_yticks((VMIN, VMAX))
    ax.plot(line_iterator, color="black")
