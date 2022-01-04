import cv2
import numpy as np

# corners: [(x,y), ...]
def extract_corner_points(corners):
    cnt = len(corners)

    differences = np.zeros((cnt, cnt), dtype=np.uint32)
    for i in range(cnt):
        for j in range(i + 1, cnt):
            dist = (corners[i][0] - corners[j][0]) ** 2 + (
                corners[i][1] - corners[j][1]
            ) ** 2

            differences[i, j] = dist
            differences[j, i] = dist

    max_index = np.unravel_index(
        np.argmax(differences, axis=None),
        differences.shape,
    )
    corner1 = max_index[0]
    corner2 = max_index[1]

    differences_added = np.zeros(cnt, dtype=np.uint32)
    for i in range(cnt):
        differences_added[i] += differences[i, corner1]
        differences_added[i] += differences[i, corner2]
    differences_added[corner1] = 0
    differences_added[corner2] = 0
    corner3 = np.argmax(differences_added, axis=None)

    for i in range(cnt):
        differences_added[i] += differences[i, corner3]
    differences_added[corner1] = 0
    differences_added[corner2] = 0
    differences_added[corner3] = 0
    corner4 = np.argmax(differences_added, axis=None)

    return [
        corners[corner1],
        corners[corner2],
        corners[corner3],
        corners[corner4],
    ]


def order_points(pts):  ## not reliable
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def transform_point(pt, matrix):
    vec = np.ones((3, 1), dtype=np.float32)
    vec[0] = pt[0]
    vec[1] = pt[1]

    transformed = np.matmul(matrix, vec)
    transformed = transformed * 1 / transformed[2]

    return transformed[0:2, :]


if __name__ == "__main__":
    inputs = np.array([[0, 2], [3, 4], [5, 1], [2, 0]], dtype=np.float32)
    outputs = np.array([[0, 2], [3, 2], [3, 0], [0, 0]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(inputs, outputs)

    print(transform_point([4, 1], M))
    print(transform_point([0, 2], M))
    print(transform_point([3, 4], M))
