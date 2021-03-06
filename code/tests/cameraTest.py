import cv2

# define a video capture object
vid = cv2.VideoCapture(1)

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow("camera", frame)

    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
cv2.destroyAllWindows()