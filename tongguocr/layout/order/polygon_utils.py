from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon


def get_rotation_matrix(img_shape: Tuple[int, int], degree: float):
    """Get the rotation matrix

    Args:
        img_shape (tuple(int, int)): Image shape
        degree (float): Rotation degree
    Returns:
        rotation matrix (np.array): Rrotation matrix in shape (2, 3)
    """
    height, width = img_shape[:2]
    if degree < 0:
        degree = 360 + degree
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)

    return matRotation


def find_center_line(poly: List[int]):
    """Gvien a unorderd polygon, find the center point of its upper and lower
    edges and also the average width of the polygon.

    Args:
        poly (list[int]): List of coordinates in shape (N, ). N must be
            even.

    Returns:
        center_points (list[float]): List of center points of upper and lower
            edges.
        avg_width (float): Average width of the polygon.
    """
    poly = np.array(poly).astype(np.float32).reshape(-1, 2)
    sort_y = np.argsort(poly[:, 1])
    top_ys = poly[sort_y[:2], 1]
    bottom_ys = poly[sort_y[-2:], 1]
    top_xs = poly[sort_y[:2], 0]
    bottom_xs = poly[sort_y[-2:], 0]

    # find the center point of upper and lower edges
    top_center_x = np.mean(top_xs)
    top_center_y = np.mean(top_ys)
    bottom_center_x = np.mean(bottom_xs)
    bottom_center_y = np.mean(bottom_ys)
    center_points = np.array(
        [top_center_x, top_center_y, bottom_center_x, bottom_center_y],
        dtype=np.float32)

    # find the average width of the polygon by calucating the area
    poly = Polygon(poly)
    avg_height = bottom_center_y - top_center_y
    avg_width = poly.area / (bottom_center_y - top_center_y)
    return center_points, avg_width, avg_height


def get_line_angle(points: List[float]):
    """Calculate the angle of a line. The angle is calculated from the
    counterclockwise rotation of the x-axis

    Args:
        points (list[float]): List of coordinates in order of
            [x1, y1, x2, y2].

    Returns:
        angle (float): Angle of the line in degree.
    """
    x1, y1, x2, y2 = points
    if x2 == x1:
        return 0.
    slope = -(y2 - y1) / (x2 - x1)
    angle = np.arctan(slope)
    if angle < 0:
        angle = -(0.5 * np.pi + angle)
    else:
        angle = 0.5 * np.pi - angle
    return angle * 180 / np.pi


def rotate_line(img_shape: Tuple[int, int], line: List[float], degree: float):
    """Rotate polygons

    Args:
        img_shape (tuple(int, int)): Image shape.
        line (list(float)): Line in shape (x_top, y_tip, x_bottom, y_bottom).
        degree (float): Rotation degree in degree.
    Returns:
        rotated line (list(float)): Rotated line.
    """
    matRotation = get_rotation_matrix(img_shape, -degree)
    reverseMatRotation = cv2.invertAffineTransform(matRotation)
    x1, y1, x2, y2 = line
    top = np.dot(reverseMatRotation, np.array([[x1], [y1], [1]]))
    bottom = np.dot(reverseMatRotation, np.array([[x2], [y2], [1]]))
    return [top[0, 0], top[1, 0], bottom[0, 0], bottom[1, 0]]


def compose_polygon(center_points: List[float], avg_width: float):
    """Given a center line and the avg_width, compose a polygon.

    Args:
        center_points (list[float]): List of center points of upper and lower
            edges.
        avg_width (float): Average width of the polygon.

    Returns:
        poly (np.float32): List of coordinates of the polygon in shape (4,2).
    """
    top_center_x, top_center_y, bot_center_x, bot_center_y = center_points
    poly = [
        top_center_x - avg_width / 2, top_center_y,
        top_center_x + avg_width / 2, top_center_y,
        bot_center_x + avg_width / 2, bot_center_y,
        bot_center_x - avg_width / 2, bot_center_y
    ]
    return np.array(poly, dtype=np.float32).reshape(-1, 2)


def shink_poly(poly: np.ndarray, avg_height: float, shrnk_ratio: float = 0.95):
    """Shrink the heigth of the polygon by a given ratio

    Args:
        poly (np.ndarray): Polygon in shape (2,2)
        avg_height (float): Average height of the polygon
        shrnk_ratio (float): Shrink ratio

    Returns:
        shrunk_poly (np.ndarray): Shrinked polygon
    """
    new_poly = poly.copy()
    shrink_height = avg_height * (1 - shrnk_ratio)
    new_poly[0, 1] += shrink_height
    new_poly[1, 1] += shrink_height
    new_poly[2, 1] -= shrink_height
    new_poly[3, 1] -= shrink_height
    return new_poly
