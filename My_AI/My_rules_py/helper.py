# Author(s): Luiz Felipe Vecchietti, Chansol Hong, Inbae Jeong
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import math

# convert degree to radian
def d2r(deg):
    return deg * math.pi / 180

# convert radian to degree
def r2d(rad):
    return rad * 180 / math.pi

# measure the distance between two coordinates (x1, y1) and (x2, y2)
def dist(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

# convert radian in (-inf, inf) range to (-PI, PI) range
def trim_radian(rad):
    adj_rad = rad
    while(adj_rad > math.pi):
        adj_rad -= 2*math.pi
    while(adj_rad < -math.pi):
        adj_rad += 2*math.pi
    return adj_rad

# clamps a value between two values in a way like modulo does
# Useful for making sure an angle is between -pi and pi
def clamp(value, min_ = -math.pi, max_ = math.pi):
    modded = (value - min_)%(max_ - min_)
    return modded + min_