# import the opencv library
import cv2
from myChessboardDetection import corner_heatmap

rows = 6
columns = 8


# define a video capture object
vid = cv2.VideoCapture(1)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # calculate the solution
    frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = corner_heatmap(frameGrey, rows, columns, 3)
    cv2.imshow("frame", corners)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()