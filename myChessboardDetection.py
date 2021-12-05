import numpy as np
import cv2

rows = 6
columns = 8


def fold_arrays(x, h):

    if len(x.shape) == 2:
        x_height, x_width = x.shape
        h_height, h_width = h.shape
        x_channels = 1
        h_channels = 1
    elif len(x.shape) > 2:
        x_height, x_width, x_channels = x.shape
        h_height, h_width, h_channels = h.shape
    else:
        raise Exception("Not enough dimensions")

    if x_channels != 1 and h_channels != 1:
        raise Exception("only one channel allowed")

    result = np.zeros_like(x, dtype=np.uint8)

    for x_y in range(x_height):
        for x_x in range(x_width):
            inter = 0
            for h_y in range(h_height):
                for h_x in range(h_width):
                    inter += (
                        x[(x_y + h_y) % x_height][(x_x + h_x) % x_width] * h[h_y][h_x]
                    )
            result[x_y, x_x] = inter

    return result


if __name__ == "__main__":
    # load image
    image = cv2.imread("./easy.png")
    # image = cv2.imread("./photo.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image is None:
        raise Exception("Image not found")

    overlay = np.array(
        [
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, 1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1],
        ]
    )

    test = fold_arrays(image, overlay)
    cv2.normalize(test, test, 0.0, 255.0, cv2.NORM_MINMAX)
    test = test.astype(np.uint8)

    # display image
    cv2.imshow("Grey", image)
    cv2.imshow("Folded", test)
    cv2.imwrite("Folded.png", test)

    # cleanup
    cv2.waitKey()
    cv2.destroyAllWindows()