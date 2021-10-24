# line_iterator.py
This is Python module to get array of pixel values on a line.

## Usage
You can create an instance of `LineIterator` class as following.
It is immutable array of pixel values on the line between `pt1` and `pt2`. 
```py
line_iterator = LineIterator(img: numpy.ndarray, pt1: Any, pt2: Any)
```

This class is subclass of `numpy.ndarray`, so that you can access its attributes like `shape` and methods like `mean()`.

Futhermore, this class has an additional method `show()`.
You can visualize its pixel values right away by using it.
BGR, RGB and gray color spaces are supported.
```py
line_iterator.show(color_space: Int)
```

## Dependency
This module depends on following external libraries:
- matplotlib (for visualization)
- numpy
- opencv-python
- scikit-image
