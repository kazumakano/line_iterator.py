import cv2
import numpy as np
import line_iterator as li
import os.path as path

img: np.ndarray = cv2.imread(path.dirname(__file__) + "/../sample.png")
# img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

line_iterator = li.LineIterator(img, (0, 0), (img.shape[0] - 20, 200))

print(f"line length is {len(line_iterator)}")

line_iterator.show()
# line_iterator.show(color_space=li.GRAY)
