import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.linalg as sla

from collections import defaultdict, Counter
from itertools import combinations
from scipy.spatial import distance

PARAMS = {3: (7, 21, 3),
          2: (5, 10, 2)}
PRECISION = 1e-4


class Circle:

    def __init__(self, dots, a, b, r):
        """
        Parameters:
            dots: set of included dots,
            (a, b): center of circle,
            r: radius
        """
        self.dots = set(dots)
        self.a = a
        self.b = b
        self.r = r

    def check(self, dot):
        """
        Check if external dot lies on circle by calculating euclidean distance from center.
        Parameters:
            dot: external  dot
        """
        return np.abs(distance.euclidean((self.a, self.b), dot) - self.r) < PRECISION

    def __len__(self):
        """
        Returns:
            number of dots
        """
        return len(self.dots)


def compare(value1, value2):
    return np.abs(value1 - value2) < PRECISION


def compare_circles(circle1, circle2):
    a1, a2 = circle1.a, circle2.a
    b1, b2 = circle1.b, circle2.b
    r1, r2 = circle1.r, circle2.b
    return compare(a1, a2) and compare(b1, b2) and compare(r1, r2)


def contains_dots(circle1, circle2):
    return len(circle1.dots & circle2.dots) > 0


def check_circle(dots):
    """
    If possible, creates circle on 3 dots, otherwise returns None
    Parameters:
        dots: 3 element list of tuples (x, y)
    Returns:
        Circle object or None
    """
    x1, y1 = dots[0]
    x2, y2 = dots[1]
    x3, y3 = dots[2]

    M = np.array([[x1 ** 2 + y1 ** 2, x1, y1, 1]
                     , [x2 ** 2 + y2 ** 2, x2, y2, 1]
                     , [x3 ** 2 + y3 ** 2, x3, y3, 1]])

    M11 = sla.det(np.delete(M, 0, axis=1))
    M12 = sla.det(np.delete(M, 1, axis=1))
    M13 = sla.det(np.delete(M, 2, axis=1))
    M14 = sla.det(np.delete(M, 3, axis=1))

    a = 0.5 * M12 / M11
    b = -0.5 * M13 / M11
    square_r = a ** 2 + b ** 2 + M14 / M11

    if square_r > 0:
        return Circle(dots, a, b, np.sqrt(square_r))

    return None


def find_fst_circle(inputData, sample, threshold):
    circles = []
    for comb in combinations(sample, 3):
        curr_circle = check_circle(comb)
        if curr_circle is not None:
            unique = True

            for circle in circles:
                if compare_circles(curr_circle, circle):
                    circle.dots |= set(comb)
                    unique = False
                    break
            if unique:
                circles.append(curr_circle)

    for dot in inputData:
        for circle in circles:
            if circle.check(dot):
                circle.dots.add(dot)
                if len(circle) >= threshold:
                    return circle
    return None


def get_circle(inputData, threshold, needed_size, iterations):
    circle = None
    if len(inputData) >= needed_size:
        for i in range(0, iterations):
            low = threshold * i
            high = threshold * (i + 1)
            circle = find_fst_circle(inputData, inputData[low: high], threshold)
            if circle is not None:
                break
    return circle


def find_max_circle(inputData, M, threshold):
    circles = []
    for comb in combinations(inputData, 3):
        curr_circle = check_circle(comb)
        if curr_circle is not None:
            unique = True

            for circle in circles:
                if compare_circles(curr_circle, circle):
                    circle.dots |= set(comb)
                    unique = False

                    if len(circle) >= threshold:
                        return circle
                    break

            if unique:
                circles.append(curr_circle)

    max_dots = 3
    max_circle = circles[0]
    all_dots = len(inputData)
    for dot in inputData:

        for circle in circles:
            if circle.check(dot):
                circle.dots.add(dot)
                if (all_dots - len(circle) == M - 1):
                    return circle

                if max_dots < len(circle) and (all_dots - len(circle) >= M - 1):
                    max_dots = len(circle)
                    max_circle = circle

                if max_dots >= threshold:
                    return max_circle

    return max_circle


def find2_bad_conditions(inputData):
    result = {}
    if len(inputData) in [2, 3]:
        i = 1
        for dot in inputData:
            result[dot] = i
            i += 1
            i = 1 if i > 2 else i
        return result

    threshold = PARAMS[2][0]
    circle = find_max_circle(inputData, 3, threshold)

    for dot in inputData:
        if circle.check(dot):
            result[dot] = 2
        else:
            result[dot] = 1
    return result


def find3_bad_conditions(inputData):
    result = defaultdict(lambda: 1)
    if len(inputData) in [3, 4]:
        i = 1
        for dot in inputData:
            result[dot] = i
            i += 1
            i = 1 if i > 3 else i
        return result

    threshold = PARAMS[3][0]
    circle = find_max_circle(inputData, 3, threshold)

    for dot in inputData:
        if circle.check(dot):
            result[dot] = 3

    return result


def solution(inputData, M):
    result = defaultdict(lambda: 1)

    if M == 1:
        return result

    inputSet = set(inputData)

    for m in range(M, 1, -1):
        if len(inputSet) == 0:
            break

        threshold, needed_size, iterations = PARAMS[m]

        if len(inputSet) < needed_size:
            if m == 3:
                result3 = find3_bad_conditions(inputSet)
                for key in result3.keys():
                    result[key] = result3[key]
                    inputSet.remove(key)
                continue

            if m == 2:
                result2 = find2_bad_conditions(inputSet)
                for key in result2.keys():
                    result[key] = result2[key]
                return result

        classified = set()
        circle = get_circle(list(inputSet), threshold, needed_size, iterations)
        for dot in inputSet:
            if circle.check(dot):
                result[dot] = m
                classified.add(dot)
        inputSet -= classified
    return result


def print_result(result, inputData):
    map_dict = defaultdict(lambda: -1)
    i = 1
    for dot in inputData:
        cls = result[dot]

        if map_dict[cls] == -1:
            map_dict[cls] = i
            i += 1

        print(map_dict[cls])


if __name__ == '__main__':

    inputData = []
    N, M = list(map(int, input().split()))
    for i in range(N):
        inputData.append(tuple(map(float, input().split())))

    result = solution(inputData, M)
    print_result(result, inputData)
