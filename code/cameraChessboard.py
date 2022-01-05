import cv2
from myChessboardDetection import corner_heatmap, normalArrayToCV2CompatibleCorners

if __name__ == "__main__":
    # define size of the checkerboard
    rows = 5
    columns = 7

    # define a video capture object
    vid = cv2.VideoCapture(1)

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()

        # calculate the solution
        use_own_solution = True
        show_processed = False
        if use_own_solution:
            # own chessboard corner detection
            title = "My Detection"
            corners, processed = corner_heatmap(frame, rows, columns, 3)

            corners = normalArrayToCV2CompatibleCorners(corners)
            ret = len(corners) == rows * columns

            if show_processed:
                frame = processed
        else:
            title = "CV2 Detection"
            # bib-solution
            ret, corners = cv2.findChessboardCorners(frame, (rows, columns), None)

        # show detection
        cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
        cv2.imshow(title, frame)

        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()
    cv2.destroyAllWindows()