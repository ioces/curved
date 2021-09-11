# Curved

Curved is a Python module written in Rust that performs curve simplification. It has been heavily optimised, and currently performs better than the simplification algorithm in [Shapely](https://pypi.org/project/Shapely/). At the moment, only the [Ramer-Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) has been implemented.

The simplification is not limited to 2D shapes, unlike traditional implementations. It is possible to simplify the points of an N dimensional shape.

## Installation

Source, and limited binary packages are provided on PyPI. These can be installed via a simple `pip install curved`. Note, if installing from a source package, make sure you have a suitable Rust build environment available.

## Usage

```python
import numpy as np
import curved

# Let's simplify a 1000 point circle down to just 33 points
t = np.linspace(0, 2 * np.pi, 1000)
points = np.vstack((np.cos(t), np.sin(t))).T
mask = curved.rdp(points, 0.01)
simplified = points[mask]
```

## Contributions

Contributions are welcome. Please feel free to create issues for bugs or feature requests. If you have code to contribute, feel free to open a pull request.