"""

Author: dizhong zhu
Date: 04/02/2021

"""


def to_left_test(p, q, s):
    """
    Test wheter point s is on the left side of the vector from p->q
    p: size=[N,2]
    q: size=[N,2]
    s: size=[N,2]
    """
    return area2(p, q, s) > 0


def area2(p, q, s):
    """
    compute the area for triangle construct by (p,q,s)
    p: size=[N,2]
    q: size=[N,2]
    s: size=[N,2]

    return: area of the triangle with size
    """
    area = p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0] + \
           q[..., 0] * s[..., 1] - q[..., 1] * s[..., 0] + \
           s[..., 0] * p[..., 1] - s[..., 1] * p[..., 0]

    return area


def in_triangle_test(p, q, r, s):
    """
    Test whether s in side the triangle (p,q,r). Triangle must be in counter clock wise. p->q->r
    p: size=[N,2]
    q: size=[N,2]
    s: size=[N,2]
    r: size=[N,2]
    """
    return to_left_test(p, q, s) & to_left_test(q, r, s) & to_left_test(r, p, s)


def in_triangle_test2(p, q, r, s):
    """
    Test whether s in side the triangle (p,q,r).
    If the triangle is CCW, then the to_left_test should be all true.
    if the triangle is CW, then the to_left_test should be all false.
    p: size=[N,2]
    q: size=[N,2]
    s: size=[N,2]
    r: size=[N,2]
    """
    res1 = to_left_test(p, q, s)
    res2 = to_left_test(q, r, s)
    res3 = to_left_test(r, p, s)

    t1 = res1 & res2 & res3
    t2 = res1 | res2 | res3

    t1 = t1 == True  # Only the true point in the CCW triangles
    t2 = t2 == False  # Only the false point in the CW triangles

    return t1 | t2

# I will implemnt barycentric later, it's very easy to do this with area 2 functions.
