import numpy as np

camera_matrix = np.array(
    [[528.6301, 0.0, 315.1405], [0.0, 532.19147, 251.70534], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# t: np.array N x 1
# G: np.array N x M
# returns: w: np.array M x 1
# t = G * w
def optimise_params(t, G):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)), G.T), t)


# corners: [(x,y), ...], needs to be sorted
def reconstruct_extrinsic_parameters(
    corners, rows, columns, cm_width_rows=10.0, cm_width_columns=10.0
):
    assert len(corners) == rows * columns

    # inverse matrix to calculate ideal camera coordinates
    inverse_camera_matrix = np.linalg.inv(camera_matrix)

    # world points
    points_in_use = (rows, columns)
    # pepare points
    objp = np.ones((points_in_use[0] * points_in_use[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : points_in_use[0], 0 : points_in_use[1]].T.reshape(-1, 2)
    # make points metric cm
    objp[:, 1] *= cm_width_columns / (points_in_use[1] - 1)
    objp[:, 0] *= cm_width_rows / (points_in_use[0] - 1)

    ideal_camera_coordinates_x = np.zeros((rows * columns, 1), dtype=np.float32)
    ideal_camera_coordinates_y = np.zeros((rows * columns, 1), dtype=np.float32)
    ideal_camera_coordinates_z = np.zeros((rows * columns, 1), dtype=np.float32)
    world_coordinates_matrix = np.ones((rows * columns, 4), dtype=np.float32)

    for i in range(rows * columns):
        cor = corners[i]
        cam_coord = np.matmul(
            inverse_camera_matrix, np.array([[cor[0]], [cor[1]], [cor[2]]])
        )
        ideal_camera_coordinates_x[i, 0] = cam_coord[0]
        ideal_camera_coordinates_y[i, 0] = cam_coord[1]
        ideal_camera_coordinates_z[i, 0] = cam_coord[2]

        world_coordinates_matrix[i, 1:4] = [objp[i][0], objp[i][1], objp[i][2]]

    t_r_1 = optimise_params(ideal_camera_coordinates_x, world_coordinates_matrix)
    t_r_2 = optimise_params(ideal_camera_coordinates_y, world_coordinates_matrix)
    t_r_3 = optimise_params(ideal_camera_coordinates_z, world_coordinates_matrix)


if __name__ == "__main__":
    reconstruct_extrinsic_parameters([(1, 2, 3), (2, 3, 4), (4, 5, 6), (7, 8, 9)], 2, 2)
