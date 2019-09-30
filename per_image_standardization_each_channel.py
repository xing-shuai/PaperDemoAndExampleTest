import tensorflow as tf
import numpy as np
import cv2

origin = cv2.resize(cv2.imread('/home/shuai/Desktop/tool/lena.jpg'), (400, 400))

a = tf.constant(origin, dtype=tf.float32)

a = tf.unstack(a, axis=2)
res = []
for i in a:
    res.append(tf.squeeze(tf.image.per_image_standardization(tf.expand_dims(i, axis=2)), axis=2))
a = tf.stack(res, axis=2)

b = tf.image.per_image_standardization(origin)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    a_, b_ = sess.run([a, b])

    # numpy 分通道标准化图像
    image = origin
    mean1, mean2, mean3 = np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2])
    std1, std2, std3 = np.std(image[:, :, 0]), np.std(image[:, :, 1]), np.std(image[:, :, 2])
    i1 = (image[:, :, 0] - mean1) / std1
    i2 = (image[:, :, 1] - mean2) / std2
    i3 = (image[:, :, 2] - mean3) / std3
    image = np.stack([i1, i2, i3], axis=2)

    cv2.imshow('tf each channel', a_)
    cv2.imshow('tf all channel', b_)
    cv2.imshow('numpy', image)
    cv2.imshow('origin', origin)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()
