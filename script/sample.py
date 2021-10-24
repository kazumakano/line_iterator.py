import cv2
import numpy as np
import line_iterator as li

IMG_FILE = "path to your image file"

def sample():
    img: np.ndarray = cv2.imread(IMG_FILE)

    line_iterator = li.LineIterator(img, (0, 0), (img.shape[0], img.shape[1]))

    print(f"line length is {len(line_iterator)}")

    line_iterator.show()

if __name__ == "main":
    sample()
