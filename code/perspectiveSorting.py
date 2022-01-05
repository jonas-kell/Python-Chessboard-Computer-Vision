import cv2
import numpy as np

# corners: [(x,y), ...]
def extract_corner_points(corners):
    cnt = len(corners)

    differences = np.zeros((cnt, cnt), dtype=np.uint64)
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

    differences_added = np.ones(cnt, dtype=np.uint64)
    for i in range(cnt):
        differences_added[i] *= differences[i, corner1]
        differences_added[i] *= differences[i, corner2]
    corner3 = np.argmax(differences_added, axis=None)

    for i in range(cnt):
        differences_added[i] *= differences[i, corner3]
    corner4 = np.argmax(differences_added, axis=None)

    return [
        corners[corner1],
        corners[corner2],
        corners[corner3],
        corners[corner4],
    ]


# corners: [(x,y), ...]
def sort_corners(corners, rows, columns):
    assert rows * columns == len(corners)

    inputs = order_points(extract_corner_points(corners))
    const = 10
    inner = int(((rows + columns) / 4) * const)
    outer = const * rows * columns - inner
    outputs = order_points(
        [
            [inner, inner],
            [outer, outer],
            [outer, inner],
            [inner, outer],
        ]
    )

    M = cv2.getPerspectiveTransform(inputs, outputs)

    transformed_corners = []  # [((x_t, y_t), (x_o, y_o)), ...]
    for cor in corners:
        transformed_corners.append((transform_point(cor, M), cor))

    number_down = 0
    for trans in transformed_corners:
        if trans[0][0] < 2 * inner:
            number_down += 1

    # find orientation
    if rows != columns:
        if number_down == rows:
            down_count = rows
            side_count = columns
        else:
            down_count = columns
            side_count = rows
        if number_down != columns and number_down != rows:
            print("Ordering-Error")
    else:
        down_count = rows
        side_count = rows
        if number_down != rows:
            print("Ordering-Error")

    sorted_corners = []
    for i in range(down_count):
        down_lower_limit = i * side_count * const  # intentionally swapped
        down_upper_limit = (i + 1) * side_count * const  # intentionally swapped
        for j in range(side_count):
            side_lower_limit = j * down_count * const  # intentionally swapped
            side_upper_limit = (j + 1) * down_count * const  # intentionally swapped
            for trans in transformed_corners:
                look_at_x = trans[0][0]
                look_at_y = trans[0][1]
                if (
                    look_at_x > side_lower_limit
                    and look_at_x < side_upper_limit
                    and look_at_y > down_lower_limit
                    and look_at_y < down_upper_limit
                ):
                    sorted_corners.append(trans[1])

    return sorted_corners


# target order: lower_left -> lower_right -> upper_right -> upper_left
def order_points(pts):
    assert len(pts) == 4

    avg_x = 0
    avg_y = 0
    for i in range(4):
        assert len(pts[i]) == 2
        avg_x += pts[i][0]
        avg_y += pts[i][1]
    avg_x /= 4
    avg_y /= 4

    angles = []
    for i in range(4):
        angles.append((np.angle(complex(pts[i][0] - avg_x, pts[i][1] - avg_y)), i))
    angles.sort()

    out = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        out[i][0] = pts[angles[i][1]][0]
        out[i][1] = pts[angles[i][1]][1]

    return out


def transform_point(pt, matrix):
    vec = np.ones((3, 1), dtype=np.float32)
    vec[0] = pt[0]
    vec[1] = pt[1]

    transformed = np.matmul(matrix, vec)
    transformed = transformed * 1 / transformed[2]

    return (transformed[0, 0], transformed[1, 0])


if __name__ == "__main__":
    inputs = order_points([[2, 0], [3, 4], [5, 1], [0, 2]])
    outputs = order_points([[0, 2], [3, 2], [3, 0], [0, 0]])

    M = cv2.getPerspectiveTransform(inputs, outputs)

    print(transform_point([4, 1], M))
    print(transform_point([0, 2], M))
    print(transform_point([3, 4], M))
