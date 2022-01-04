import numpy as np
import cv2
from numpy.core.defchararray import index

from graphOperations import filter_by_graph_method
from perspectiveSorting import extract_corner_points

rows = 6
columns = 8


def generate_mask_array(template, spread):
    assert template.shape[0] == template.shape[1]
    assert int(spread) == spread

    template_size = template.shape[0]
    new_size = template_size * spread

    # map the filters but spread out their range
    new_mask = np.zeros((new_size, new_size), dtype=np.int8)
    new_mask[0 : new_size // 2 : spread, 0 : new_size // 2 : spread] = template[
        0 : template_size // 2 : 1, 0 : template_size // 2 : 1
    ]
    new_mask[-1 : -1 - new_size // 2 : -spread, 0 : new_size // 2 : spread] = template[
        -1 : -1 - template_size // 2 : -1, 0 : template_size // 2 : 1
    ]
    new_mask[0 : new_size // 2 : spread, -1 : -1 - new_size // 2 : -spread] = template[
        0 : template_size // 2 : 1, -1 : -1 - template_size // 2 : -1
    ]
    new_mask[
        -1 : -1 - new_size // 2 : -spread, -1 : -1 - new_size // 2 : -spread
    ] = template[-1 : -1 - template_size // 2 : -1, -1 : -1 - template_size // 2 : -1]

    return new_mask


def normalArrayToCV2CompatibleCorners(array):
    length = len(array)
    assert length > 0
    tuple_length = len(array[0])
    assert tuple_length == 2

    result = np.zeros((length, 1, 2), dtype=np.float32)
    result[:, 0, ::-1] = array

    return result


def corner_heatmap(image, rows, columns, spread=1):
    if len(image.shape) == 2:
        image_channels = 1
    elif len(image.shape) > 2:
        _, _, image_channels = image.shape
    else:
        raise Exception("Not enough dimensions")
    spread = int(spread)
    if spread < 1:
        raise Exception("Spread needs to be greater than 1")
    if rows < 2:
        raise Exception("rows need to be at least 2")
    if columns < 2:
        raise Exception("columns need to be at least 2")

    m_c_template = np.array(
        [
            [00, 00, +1, +1, -1, -1, 00, 00],
            [00, +1, +1, +1, -1, -1, -1, 00],
            [+1, +1, +1, +1, -1, -1, -1, -1],
            [+1, +1, +1, 00, 00, -1, -1, -1],
            [-1, -1, -1, 00, 00, +1, +1, +1],
            [-1, -1, -1, -1, +1, +1, +1, +1],
            [00, -1, -1, -1, +1, +1, +1, 00],
            [00, 00, -1, -1, +1, +1, 00, 00],
        ],
        dtype=np.int8,
    )
    m_e_template = np.array(
        [
            [00, +1, +1, +1, +1, +1, +1, 00],
            [-1, 00, +1, +1, +1, +1, 00, -1],
            [-1, -1, 00, +1, +1, 00, -1, -1],
            [-1, -1, -1, 00, 00, -1, -1, -1],
            [-1, -1, -1, 00, 00, -1, -1, -1],
            [-1, -1, 00, +1, +1, 00, -1, -1],
            [-1, 00, +1, +1, +1, +1, 00, -1],
            [00, +1, +1, +1, +1, +1, +1, 00],
        ],
        dtype=np.int8,
    )
    m_v_template = np.array(
        [
            [00, +1, +1, +1, +1, +1, +1, 00],
            [00, 00, +1, +1, +1, +1, 00, 00],
            [00, 00, 00, +1, +1, 00, 00, 00],
            [00, 00, 00, 00, 00, 00, 00, 00],
            [00, 00, 00, 00, 00, 00, 00, 00],
            [00, 00, 00, -1, -1, 00, 00, 00],
            [00, 00, -1, -1, -1, -1, 00, 00],
            [00, -1, -1, -1, -1, -1, -1, 00],
        ],
        dtype=np.int8,
    )
    m_h_template = np.array(
        [
            [00, 00, 00, 00, 00, 00, 00, 00],
            [-1, 00, 00, 00, 00, 00, 00, +1],
            [-1, -1, 00, 00, 00, 00, +1, +1],
            [-1, -1, -1, 00, 00, +1, +1, +1],
            [-1, -1, -1, 00, 00, +1, +1, +1],
            [-1, -1, 00, 00, 00, 00, +1, +1],
            [-1, 00, 00, 00, 00, 00, 00, +1],
            [00, 00, 00, 00, 00, 00, 00, 00],
        ],
        dtype=np.int8,
    )
    m_d_template = np.array(
        [
            [00, 00, -1, -1, 00, 00, 00, 00],
            [00, -1, -1, -1, 00, 00, 00, 00],
            [-1, -1, -1, -1, 00, 00, 00, 00],
            [-1, -1, -1, 00, 00, 00, 00, 00],
            [00, 00, 00, 00, 00, +1, +1, +1],
            [00, 00, 00, 00, +1, +1, +1, +1],
            [00, 00, 00, 00, +1, +1, +1, 00],
            [00, 00, 00, 00, +1, +1, 00, 00],
        ],
        dtype=np.int8,
    )
    m_u_template = np.array(
        [
            [00, 00, 00, 00, +1, +1, 00, 00],
            [00, 00, 00, 00, +1, +1, +1, 00],
            [00, 00, 00, 00, +1, +1, +1, +1],
            [00, 00, 00, 00, 00, +1, +1, +1],
            [-1, -1, -1, 00, 00, 00, 00, 00],
            [-1, -1, -1, -1, 00, 00, 00, 00],
            [00, -1, -1, -1, 00, 00, 00, 00],
            [00, 00, -1, -1, 00, 00, 00, 00],
        ],
        dtype=np.int8,
    )

    m_c = generate_mask_array(m_c_template, spread)
    m_e = generate_mask_array(m_e_template, spread)
    m_v = generate_mask_array(m_v_template, spread)
    m_h = generate_mask_array(m_h_template, spread)
    m_d = generate_mask_array(m_d_template, spread)
    m_u = generate_mask_array(m_u_template, spread)
    h_dimension = m_u.shape[0]  # counts for all masks

    if image_channels == 1:  # greyscale
        # blur parameter needs to be un-even
        blurred = cv2.medianBlur(image, spread + 1 if spread % 2 == 0 else spread)
        msk = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:  # color
        # blur parameter needs to be un-even
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, spread + 1 if spread % 2 == 0 else spread)
        # OpenCV uses H: 0-179, S: 0-255, V: 0-255
        lwr = np.array([0, 0, 135])
        upr = np.array([179, 120, 255])
        msk = cv2.inRange(hsv, lwr, upr)  # filter for white

    workImage = np.where(msk > 0, 1, -1).astype(np.int16)
    # np.int16 is a very important type here, to force the convolution output to be signed

    highlight_corner = cv2.filter2D(
        workImage, -1, m_c, anchor=(h_dimension // 2, h_dimension // 2)
    )
    highlight_edges = cv2.filter2D(
        workImage, -1, m_e, anchor=(h_dimension // 2, h_dimension // 2)
    )
    highlight_vertical = cv2.filter2D(
        workImage, -1, m_v, anchor=(h_dimension // 2, h_dimension // 2)
    )
    highlight_horizontal = cv2.filter2D(
        workImage, -1, m_h, anchor=(h_dimension // 2, h_dimension // 2)
    )
    highlight_down = cv2.filter2D(
        workImage, -1, m_d, anchor=(h_dimension // 2, h_dimension // 2)
    )
    highlight_up = cv2.filter2D(
        workImage, -1, m_u, anchor=(h_dimension // 2, h_dimension // 2)
    )
    highlight = (
        np.abs(highlight_corner)
        + np.abs(highlight_edges)
        - np.abs(highlight_vertical)
        - np.abs(highlight_horizontal)
        - np.abs(highlight_down)
        - np.abs(highlight_up)
    )

    # cv2.imshow(
    #     "Pre-processed", 100 * (1 + workImage.astype(np.uint8))
    # )  # can be deactivated
    # cv2.imshow("highlight", highlight.astype(np.uint8))  # can be deactivated

    mask_size = int(h_dimension / 1.5)  # assumption of what is necessary to cover
    corner_candidates = []  # [(x,y),...]
    number_of_candidates_to_sample = int(
        (rows - 1) * (columns - 1) * 1.3
    )  # give the algorithm a bit of leeway
    min_additional_detection_value = 30
    max_corners_to_detect = (rows - 1) * (columns - 1)
    for i in range(number_of_candidates_to_sample):
        max_index = np.unravel_index(
            np.argmax(highlight, axis=None),
            highlight.shape,
        )
        if (
            i >= max_corners_to_detect
            and highlight[max_index[0], max_index[1]] <= min_additional_detection_value
        ):
            break  # stop sampling
        corner_candidates.append((max_index[0], max_index[1]))
        highlight[
            max(0, max_index[0] - mask_size) : min(
                max_index[0] + mask_size, highlight.shape[0]
            ),
            max(0, max_index[1] - mask_size) : min(
                max_index[1] + mask_size, highlight.shape[1]
            ),
        ] = 0

    return (
        extract_corner_points(
            extract_sorted_corners_form_candidates_graph(corner_candidates)
        ),
        # ((workImage + 1) * 100).astype(np.uint8),
        (highlight).astype(np.uint8),
    )


# corner_candidates: # [(x,y),...]
def extract_sorted_corners_form_candidates_graph(corner_candidates):
    if len(corner_candidates) == 0:
        return [(0, 0)]

    if (rows - 1) * (columns - 1) >= len(corner_candidates):
        # too little corners detected for further filtering. Need to return all
        return corner_candidates

    av_x = 0
    av_y = 0
    for cor in corner_candidates:
        av_x += cor[0]
        av_y += cor[1]
    av_x = int(av_x / len(corner_candidates))
    av_y = int(av_y / len(corner_candidates))

    return filter_by_graph_method(
        corner_candidates, (av_x, av_y), (rows - 1) * (columns - 1)
    )


if __name__ == "__main__":
    # load image
    # image = cv2.imread("./easy.png")
    # image = cv2.imread("./easy30.png")
    # image = cv2.imread("./easy45.png")
    # image = cv2.imread("./photo.png")
    image = cv2.imread("./photo45.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image is None:
        raise Exception("Image not found")

    corners, processed = corner_heatmap(image, rows, columns, 6)

    # display image
    # cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Corners", (300, 600))
    imageWithCorners = cv2.drawChessboardCorners(
        image,
        (rows - 1, columns - 1),
        normalArrayToCV2CompatibleCorners(corners),
        True,
    )
    cv2.imshow("Corners", processed)

    # cleanup
    cv2.waitKey()
    cv2.destroyAllWindows()

    # compare with library
    ret, corners = cv2.findChessboardCorners(image, (rows - 1, columns - 1), None)

    if ret == False:
        cv2.drawChessboardCorners(image, (rows - 1, columns - 1), corners, ret)
        # cv2.namedWindow("cv2:", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("cv2:", (300, 600))
        cv2.imshow("cv2:", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print("CV2 is bad...")