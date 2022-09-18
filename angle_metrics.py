import cv2
import numpy as np
from math import sqrt, degrees, atan


def get_dist(coords):
    x1, y1, x2, y2 = (
        coords[0],
        coords[1],
        coords[2],
        coords[3],
    )
    dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return (dist, [x1, y1, x2, y2])


def calc_angle(coords):
    x1, y1, x2, y2 = (
        coords[1][0],
        coords[1][1],
        coords[1][2],
        coords[1][3],
    )
    if y1 > y2:
        slope = (y1 - y2) / (x1 - x2)
    else:
        slope = (y2 - y1) / (x2 - x1)

    angle = degrees(atan(slope))
    return angle


def get_skii_angle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 50
    max_line_gap = 20
    line_image = np.copy(img) * 0

    lines = cv2.HoughLinesP(
        edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )

    lengths = []
    if lines is not None:
        for line in lines:
            lengths.append(get_dist(line[0]))
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        return calc_angle(max(lengths))
    else:
        return -1
