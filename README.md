# line_iterator.py
This is Python module to get array of pixel values on a line.

# Usage
You can create an instance of `LineIterator` class as following.
It is array of pixel values on the line between `pt1` and `pt2`. 
```py
line_iterator = LineIterator(img: numpy.ndarray, pt1: Any, pt2: Any)
```

`LineIterator` is subclass of `numpy.ndarray`, so that you can access its attributes like `shape` and methods like `mean()`.
In addition, `LineIterator` class has method `show()`.
You can visualize pixel values right away by using it.
```py
line_iterator.show()
```
