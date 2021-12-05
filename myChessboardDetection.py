import numpy as np
import cv2

rows = 6
columns = 8

# load image
image = cv2.imread("./photo.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# display image
cv2.imshow("Grey", image)

# cleanup
cv2.waitKey()
cv2.destroyAllWindows()