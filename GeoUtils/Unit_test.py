"""

Author: dizhong zhu
Date: 04/02/2021

"""

import numpy as np

from GeoUtils.Geo2D.common2D import (
    in_triangle_test
)


# Add unit test here
def test_in_triangle():
    triangles = np.array([[[0, 0], [1, 0], [0, 1]],
                          [[0, 0], [1, 0], [0, 1]],
                          [[0, 0], [1, 0], [0, 1]]])

    s = np.array([[0.5, 0.5],
                  [0.3, 0.5],
                  [1, 1]])

    expect_res = [False, True, False]

    p = triangles[:, 0, :]
    q = triangles[:, 1, :]
    r = triangles[:, 2, :]

    res = in_triangle_test(p, q, r, s)

    if np.any((res == expect_res) == False):
        print('{0} failed!'.format(test_in_triangle.__name__))
    else:
        print('{0} test passed!'.format(test_in_triangle.__name__))


if __name__ == '__main__':
    test_in_triangle()
