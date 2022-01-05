import cv2
import numpy as np

# cm_width_rows: distance in cm from the outer most CORNER-POINTS of rows
# cm_width_columns: distance in cm from the outer most CORNER-POINTS of columns
def calibrate_from_checkerboard_images(
    images, rows, columns, cm_width_rows=10.0, cm_width_columns=10.0
):
    points_in_use = (rows, columns)

    # pepare points
    objp = np.zeros((points_in_use[0] * points_in_use[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : points_in_use[0], 0 : points_in_use[1]].T.reshape(-1, 2)

    # make points metric cm
    objp[:, 1] *= cm_width_columns / (points_in_use[1] - 1)
    objp[:, 0] *= cm_width_rows / (points_in_use[0] - 1)

    # arrays to store points from the image
    imgpoints = []  # 2d points in image plane
    objpoints = []  # 3d points in real world space

    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.0001)

    # iterate over images
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the corners (bib function for format)
        ret, corners = cv2.findChessboardCorners(gray, points_in_use, None)

        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if False:  # enable to check the frames
                cv2.drawChessboardCorners(img, points_in_use, corners2, ret)
                cv2.imshow("check", img)
                cv2.waitKey()
                cv2.destroyAllWindows()

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


if __name__ == "__main__":
    number_of_frames_to_use = 40
    every_nth_frame = 10

    # define a video capture object
    vid = cv2.VideoCapture(1)

    frames = []
    i = 0
    while True:
        i += 1
        ret, frame = vid.read()

        if ret:
            cv2.imshow("Cam", frame)
            cv2.waitKey(1)

        if ret and i % every_nth_frame == 0:
            frames.append(frame)
            print(str(int(i / every_nth_frame)) + "/" + str(number_of_frames_to_use))

        if not ret or i > every_nth_frame * number_of_frames_to_use:
            break

    # After the loop release the cap object
    vid.release()
    cv2.destroyAllWindows()

    output = calibrate_from_checkerboard_images(frames, 5, 7, 10.2, 15.3)
    ret, mtx, dist, rvecs, tvecs = output

    print("Ret")
    print(ret)

    print("Mtx")
    print(mtx)

    # print("Dist")
    # print(dist)

    # print("rvecs")
    # print(rvecs)

    # print("tvecs")
    # print(tvecs)