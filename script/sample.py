from typing import Union
import cv2
import numpy as np
from line_iterator import LineIterator


def sample(file: str):
    img: Union[np.ndarray, None] = cv2.imread(file)
    if img is None:
        raise Exception(f"{file} can't be loaded")

    line_iterator = LineIterator(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1))

    print(f"line length is {len(line_iterator)}")

    line_iterator.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("img_file", help="specify path to your image file")
    img_file: str = parser.parse_args().img_file

    sample(img_file)
