import numpy as np
import cv2

rows = 6
columns = 8


def corner_heatmap(image, rows, columns, spread=1):
    if len(image.shape) == 2:
        x_height, x_width = image.shape
        x_channels = 1
    elif len(image.shape) > 2:
        x_height, x_width, x_channels = image.shape
    else:
        raise Exception("Not enough dimensions")
    if x_channels != 1:
        raise Exception("only one channel allowed")
    spread = int(spread)
    if spread < 1:
        raise Exception("Spread needs to be greater than 1")
    if rows < 2:
        raise Exception("rows need to be at least 2")
    if columns < 2:
        raise Exception("columns need to be at least 2")

    h = np.array(
        [
            [1, 1, 1, -1, -1, -1],
            [1, 1, 1, -1, -1, -1],
            [1, 1, 1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1],
            [-1, -1, -1, 1, 1, 1],
            [-1, -1, -1, 1, 1, 1],
        ]
    )
    h2 = np.array(
        [
            [1, -1, -1, -1, -1, -1],
            [1, 1, -1, -1, -1, 1],
            [1, 1, 1, -1, 1, 1],
            [1, 1, -1, 1, 1, 1],
            [1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, 1],
        ]
    )
    h_height, h_width = h.shape
    h2_height, h2_width = h2.shape

    workImage = cv2.medianBlur(image, 7)
    average = np.average(workImage)
    workImage = np.where(workImage > average, 1, 0).astype(np.uint8)

    highlight = np.zeros_like(workImage, dtype=np.uint8)

    height_range = range(x_height - max(h_height, h2_height) * spread)
    width_range = range(x_width - max(h_width, h2_width) * spread)
    for x_y in height_range:
        for x_x in width_range:
            highlight[x_y + h_height // 2, x_x + h_width // 2] += abs(
                np.sum(
                    workImage[
                        x_y : x_y + spread * h_height : spread,
                        x_x : x_x + spread * h_width : spread,
                    ]
                    * h[:, :]
                )
            )
            highlight[x_y + h2_height // 2, x_x + h2_width // 2] += abs(
                np.sum(
                    workImage[
                        x_y : x_y + spread * h2_height : spread,
                        x_x : x_x + spread * h2_width : spread,
                    ]
                    * h2[:, :]
                )
            )

    result = np.zeros_like(workImage, dtype=np.uint8)
    mask_size = max(h_height, h_width, h2_height, h2_width) * spread
    for corner in range((rows - 1) * (columns - 1)):
        max_index = np.unravel_index(np.argmax(highlight, axis=None), highlight.shape)
        highlight[
            max_index[0] - mask_size : max_index[0] + mask_size,
            max_index[1] - mask_size : max_index[1] + mask_size,
        ] = 0
        result[
            max_index[0] - 1 : max_index[0] + 1,
            max_index[1] - 1 : max_index[1] + 1,
        ] = 255

    cv2.normalize(workImage, workImage, 0.0, 255.0, cv2.NORM_MINMAX).astype(np.uint8)
    return result, workImage


if __name__ == "__main__":
    # load image
    # image = cv2.imread("./easy.png")
    # image = cv2.imread("./easy30.png")
    # image = cv2.imread("./easy45.png")
    # image = cv2.imread("./photo.png")
    image = cv2.imread("./photo45.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image is None:
        raise Exception("Image not found")

    corners, image = corner_heatmap(image, rows, columns, 3)

    # display image
    cv2.imshow("Grey", image)
    cv2.imshow("Folded", corners)

    # cleanup
    cv2.waitKey()
    cv2.destroyAllWindows()