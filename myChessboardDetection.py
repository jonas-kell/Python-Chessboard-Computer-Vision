import numpy as np
import cv2

rows = 6
columns = 8


def corner_heatmap(image, rows, columns, spread=1):
    if len(image.shape) == 2:
        image_channels = 1
    elif len(image.shape) > 2:
        _, _, image_channels = image.shape
    else:
        raise Exception("Not enough dimensions")
    if image_channels != 1:
        raise Exception("Only one channel allowed")
    spread = int(spread)
    if spread < 1:
        raise Exception("Spread needs to be greater than 1")
    if rows < 2:
        raise Exception("rows need to be at least 2")
    if columns < 2:
        raise Exception("columns need to be at least 2")

    h1 = np.array(
        [
            [1, 1, 1, 0, 0, -1, -1, -1],
            [1, 1, 1, 0, 0, -1, -1, -1],
            [1, 1, 0, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 1, 1],
            [-1, -1, -1, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    h2 = np.array(
        [
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [-1, 0, 0, 1, 1, 0, 0, -1],
            [-1, -1, -1, 0, 0, -1, -1, -1],
            [-1, -1, -1, 0, 0, -1, -1, -1],
            [-1, 0, 0, 1, 1, 0, 0, -1],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    h1_height, h1_width = h1.shape
    h2_height, h2_width = h2.shape

    workImage = cv2.medianBlur(image, 5)
    average = np.average(workImage)

    workImage = np.where(workImage > average, 1, 0).astype(np.int16)
    # very important type, to force the convolution output to be signed

    highlight1 = cv2.filter2D(workImage, -1, h1, anchor=(h1_height // 2, h1_width // 2))
    highlight2 = cv2.filter2D(workImage, -1, h2, anchor=(h2_height // 2, h2_width // 2))
    highlight = np.abs(highlight1) + np.abs(highlight2)

    cv2.imshow("Pre-processed", 240 * workImage.astype(np.uint8))  # can be deactivated
    cv2.imshow("highlight", 8 * highlight.astype(np.uint8))  # can be deactivated

    result = np.zeros_like(workImage, dtype=np.uint8)
    highlight_overwritable = highlight.copy()
    mask_size = 10
    corners = [(0, 0)] * (rows - 1) * (columns - 1)
    for i in range((rows - 1) * (columns - 1)):
        max_index = np.unravel_index(
            np.argmax(highlight_overwritable, axis=None), highlight_overwritable.shape
        )
        highlight_overwritable[
            max(0, max_index[0] - mask_size) : min(
                max_index[0] + mask_size, highlight_overwritable.shape[0]
            ),
            max(0, max_index[1] - mask_size) : min(
                max_index[1] + mask_size, highlight_overwritable.shape[1]
            ),
        ] = 0
        corners[i] = max_index

        result[
            max(0, max_index[0] - 1) : min(max_index[0] + 1, result.shape[0]),
            max(0, max_index[1] - 1) : min(max_index[1] + 1, result.shape[1]),
        ] = 255

    return result


if __name__ == "__main__":
    # load image
    # image = cv2.imread("./easy.png")
    # image = cv2.imread("./easy30.png")
    # image = cv2.imread("./easy45.png")
    image = cv2.imread("./photo.png")
    # image = cv2.imread("./photo45.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image is None:
        raise Exception("Image not found")

    corners = corner_heatmap(image, rows, columns, 3)

    # display image
    cv2.imshow("Grey", image)
    cv2.imshow("Folded", corners)

    # cleanup
    cv2.waitKey()
    cv2.destroyAllWindows()