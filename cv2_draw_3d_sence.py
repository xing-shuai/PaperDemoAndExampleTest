import cv2
import numpy as np


def generate_grid_mesh(start, end, step=1.0):
    num_point_per_line = int((end - start) // step + 1)
    its = np.linspace(start, end, num_point_per_line)
    line = []
    color = []
    common_line_color = [192, 192, 192]
    for i in range(num_point_per_line):
        line.append([its[0], its[i], 0, its[-1], its[i], 0])
        if its[i] == 0:
            color.append([0, 255, 0])
        else:
            color.append(common_line_color)

    for i in range(num_point_per_line):
        line.append([its[i], its[-1], 0, its[i], its[0], 0])
        if its[i] == 0:
            color.append([0, 0, 255])
        else:
            color.append(common_line_color)

    return np.array(line, dtype=np.float32), np.array(color, dtype=np.uint8)


def euclidean_to_homogeneous(points):
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    else:
        raise TypeError("Works only with numpy arrays")


def homogeneous_to_euclidean(points):
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    else:
        raise TypeError("Works only with numpy arrays")


def projection_to_2d_plane(vertices, view_matrix, projection_matrix, scale):
    vertices = (homogeneous_to_euclidean(
        (euclidean_to_homogeneous(vertices) @ view_matrix.T) @ projection_matrix.T)[:, :2]) * scale

    vertices[:, 1] = scale - vertices[:, 1]
    vertices[:, 0] = vertices[:, 0] + scale
    return vertices.astype(np.int32)


grid_vertices, grid_color = generate_grid_mesh(-4, 4, step=0.5)
grid_vertices = grid_vertices.reshape(-1, 3)
frame = np.zeros([800, 800])

rorate_x_90 = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]], dtype=np.float32)

grid_vertices = grid_vertices @ rorate_x_90[:3, :3].T
grid_vertices = grid_vertices @ np.array([[650, 0, 0],
                                          [0, 650, 0],
                                          [0, 0, 650]], dtype=np.float32).T

view_matrix = np.array([[0.818994, 0, -0.573802, 241.247],
                        [-0.148507, 0.965927, -0.211966, -70.451],
                        [0.554251, 0.258813, 0.791089, -9396.55],
                        [0, 0, 0, 1]], dtype=np.float32)

projection_matrix = np.array([[2.41421, 0, 0, 0],
                              [0, 2.41421, 0, 0],
                              [0, 0, -1, -0.2],
                              [0, 0, -1, 0]], dtype=np.float32)

grid_vertices = projection_to_2d_plane(grid_vertices, view_matrix, projection_matrix, 400).reshape(-1, 4)

camera_vertices = np.array([[0, 0, 0], [-1, -1, 2],
                            [0, 0, 0], [1, 1, 2],
                            [0, 0, 0], [1, -1, 2],
                            [0, 0, 0], [-1, 1, 2],
                            [-1, 1, 2], [-1, -1, 2],
                            [-1, -1, 2], [1, -1, 2],
                            [1, -1, 2], [1, 1, 2],
                            [1, 1, 2], [-1, 1, 2]], dtype=np.float32)

camera_config1 = [
    [[[-0.9115695, 0.41064942, 0.02020282], [0.06090775, 0.18347366, -0.9811359], [-0.4066096, -0.893143, -0.19226073]],
     [[-82.702095], [552.18964], [5557.3535]]],
    [[[0.93101627, 0.3647627, 0.01252435], [0.08939715, -0.19463754, -0.97679293],
      [-0.3538599, 0.91052973, -0.21381946]], [[-209.06287], [375.06915], [5818.277]]],
    [[[-0.92090756, -0.38473552, -0.06251254], [-0.02568138, 0.21992028, -0.97517973],
      [0.38893405, -0.89644504, -0.21240678]], [[623.0986], [290.9053], [5534.379]]],
    [[[0.9276671, -0.36360627, 0.08499598], [-0.01666269, -0.26770413, -0.9633571],
      [0.37303644, 0.89225835, -0.25439897]], [[-178.367], [423.46698], [4421.645]]]
]

human36m_connectivity_dict = [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12),
                              (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]

gt_skeleton = np.load('resource/gt.npy')[6]

gt_skeleton = gt_skeleton.reshape(-1, 3)

gt_skeleton = gt_skeleton @ rorate_x_90[:3, :3].T
gt_skeleton = gt_skeleton @ np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]], dtype=np.float32).T
gt_skeleton = projection_to_2d_plane(gt_skeleton, view_matrix, projection_matrix, 400).reshape(
    17, 2)

while True:
    frame = np.zeros([800, 800, 3])

    # draw line
    for index, line in enumerate(grid_vertices):
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), grid_color[index].tolist())

    # draw camera
    for conf in camera_config1:
        m_rt = np.eye(4, dtype=np.float32)
        r = np.array(conf[0], dtype=np.float32).T
        m_rt[:-1, -1] = -r @ np.array(conf[1], dtype=np.float32).squeeze()
        m_rt[:-1, :-1] = r

        m_s = np.eye(4, dtype=np.float32) * 250
        m_s[3, 3] = 1

        camera_vertices_convert = homogeneous_to_euclidean(
            euclidean_to_homogeneous(camera_vertices) @ (rorate_x_90 @ m_rt @ m_s).T)

        camera_vertices_convert = projection_to_2d_plane(camera_vertices_convert, view_matrix, projection_matrix, 400)
        camera_vertices_convert = camera_vertices_convert.reshape(-1, 4)
        for index, line in enumerate(camera_vertices_convert):
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 153, 255), thickness=1)

    for c in human36m_connectivity_dict:
        cv2.line(frame, (*gt_skeleton[c[0]],), (*gt_skeleton[c[1]],), (100, 155, 255), thickness=2)
        cv2.circle(frame, (*gt_skeleton[c[0]],), 3, (0, 0, 255), -1)
        cv2.circle(frame, (*gt_skeleton[c[1]],), 3, (0, 0, 255), -1)

    cv2.imshow("demo", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
