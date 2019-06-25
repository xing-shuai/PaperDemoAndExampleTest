import numpy as np
from HPE_heatmap_generation import CenterGaussianHeatMap


def spatial_softmax(img):
    exp_img = np.exp(img)
    exp_sum = np.sum(exp_img)
    return exp_img / exp_sum


def soft_argmax(array):
    max_position = np.zeros([2])
    r, c = np.shape(array)
    for i in range(r):
        for j in range(c):
            max_position += np.array([i, j]) * array[i, j]
    return max_position


if __name__ == "__main__":
    img = np.zeros([100, 20])

    height, width = np.shape(img)
    cy, cx = height / 2.0, width / 2.0

    heatmap = CenterGaussianHeatMap(height, width, cx, cy, 2)

    soft_heatmap = spatial_softmax(heatmap)

    print(np.unravel_index(soft_heatmap.argmax(), soft_heatmap.shape))

    print(soft_argmax(soft_heatmap))

    pass
