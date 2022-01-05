import cv2
from myChessboardDetection import corner_heatmap, normalArrayToCV2CompatibleCorners

if __name__ == "__main__":
    rows = 6
    columns = 8

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
            corners, processed = corner_heatmap(frame, rows, columns, 3)

            corners = normalArrayToCV2CompatibleCorners(corners)
            ret = len(corners) == ((rows - 1) * (columns - 1))

            if show_processed:
                frame = processed
        else:
            # bib-solution
            ret, corners = cv2.findChessboardCorners(
                frame, (rows - 1, columns - 1), None
            )

        # show detection
        cv2.drawChessboardCorners(frame, (rows - 1, columns - 1), corners, ret)
        cv2.imshow("Detection", frame)

        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()
    cv2.destroyAllWindows()