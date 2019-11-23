# # import numpy as np
# #
# # #
# # # # beta = 12
# # # # y_est = np.array([[1.1, 3.0, 1.1, 1.3, 0.8]])
# # # # a = np.exp(beta * y_est)
# # # # b = np.sum(np.exp(beta * y_est))
# # # # softmax = a / b
# # # # max = np.sum(softmax * y_est)
# # # # print(max)
# # # # pos = range(y_est.size)
# # # # print(pos)
# # # # softargmax = np.sum(softmax * pos)
# # # # print(softargmax)
# # #
# # # a = np.random.randn(100, 100) * 50
# # # a = a + -np.min(a)
# # #
# # # a = np.exp(a) / np.sum(np.exp(a))
# # # # print(np.sum(a))
# # # sum_r = np.sum(a, axis=1)
# # # sum_c = np.sum(a, axis=0)
# # #
# # # # print(a)
# # #
# # # # print(sum_c)
# # #
# # # # print(sum_c * np.arange(0, sum_c.shape[0]))
# # #
# # # print(np.sum(sum_c * np.arange(0, sum_c.shape[0])))
# # #
# # # print(np.argmax(a))
# # #
# # # print(np.unravel_index(np.argmax(a), a.shape))
# # #
# # # print(np.sum(np.reshape(a, (a.shape[0] * a.shape[1],)) * np.arange(0, a.shape[0] * a.shape[1])))
# #
# #
# # # import numpy as np
# #
# # # ss = np.zeros([64])
# # # count = 0
# # # cc = []
# # # for i in range(64):
# # #     aa = np.random.randint(0, 63)
# # #     if aa in cc:
# # #         pass
# # #     else:
# # #         cc.append(aa)
# # #
# # #         count += 1
# # #
# # # print(count / 64)
# #
# # m = np.array([[1, 2, 3, 5, 7, 6],
# #               [2, 1, 4, 5, 7, 4],
# #               [3, 4, 5, 6, 3, 6],
# #               [2, 3, 1, 4, 6, 8],
# #               [5, 6, 1, 4, 6, 2],
# #               [4, 2, 4, 1, 1, 6]])
# #
# # position = (-1, -1)
# # direction = 0
# #
# #
# # def find_ne(p, d):
# #     if d == 1:
# #         if p[1] >= 4:
# #             return ((p[0] + 2, p[1], 2, m[p[0] + 1, p[1]]))
# #         else:
# #             return ((p[0] + 2, p[1], 2, m[p[0] + 1, p[1]]), (p[0], p[1] + 2, 1, m[p[0], p[1] + 1]))
# #     if d == 0:
# #         if p[1] <= 1:
# #             return ((p[0] + 2, p[1], 2, m[p[0] + 1, p[1]]), (p[0], p[1] - 2, 0, m[p[0], p[1] - 1]))
# #         else:
# #             return ((p[0] + 2, p[1], 2, m[p[0] + 1, p[1]]))
# #     if d == 2:
# #         if p[0] >= 4:
# #
# #             return ()
# #         else:
# #             if p[1] <= 1:
# #                 return ((p[0] + 2, p[1], 2, m[p[0] + 1, p[1]]), (p[0], p[1] + 2, 1, m[p[0], p[1] + 1]))
# #             else:
# #                 if p[1] >= 4:
# #                     return ((p[0] + 2, p[1], 2, m[p[0] + 1, p[1]]), (p[0], p[1] - 2, 0, m[p[0], p[1] - 1]))
# #                 else:
# #                     return (
# #                         (p[0] + 2, p[1], 2, m[p[0] + 1, p[1]]), (p[0], p[1] - 2, 0, m[p[0], p[1] - 1]),
# #                         (p[0], p[1] + 2, 1, m[p[0], p[1] + 1]))
# #
# #
# # time = [0]
# # pre_time = 0
#
#
# import cv2 as cv
# import numpy as np
#
#
# def face_detect_demo(image):
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     face_detector = cv.CascadeClassifier("E:/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml")
#     faces = face_detector.detectMultiScale(gray, 1.1, 2)
#     for x, y, w, h in faces:
#         cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     cv.imshow("result", image)
#
#
# print("--------- Python OpenCV Tutorial ---------")
# capture = cv.VideoCapture(0)
# cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
# while (True):
#     ret, frame = capture.read()
#     frame = cv.flip(frame, 1)
#     face_detect_demo(frame)
#     c = cv.waitKey(10)
#     if c == 27:  # ESC
#         break
# cv.waitKey(0)
#
# cv.destroyAllWindows()
# #
# #
# # def step(p, d):
# #     global pre_time
# #     ne = find_ne(p, d)
# #     if len(ne) == 0:
# #         time.append(pre_time)
# #     for i in ne:
# #         pre_time = time[-1]
# #         time[-1] += i[3]
# #         step((i[0], i[1]), i[2])
# #         pass
# #     pass
# #
# #
# # for i in range(6):
# #     step((0, 0), 1)
# #
# # print(min(time))


# import os
#
# ava_ips = []
# allocated_ips = []
# for i in range(2, 255):
#     command = os.system('ping -c 1 10.21.243.' + str(i))
#     if command == 0:
#         allocated_ips.append(i)
#         pass  # Sucess
#     else:
#         ava_ips.append(i)
#         # print(str(i))
#
# print('allocated_ips', allocated_ips)
# print('ava_ips', ava_ips)

import numpy as np

aaaa = [1, 2, [2.0, 4.0], ['sdssds'], 'sdsdsd', np.random.random((100, 100)),
        [np.random.random((100, 100)), np.random.random((100, 100))]]
aaaa.remove(aaaa[-1])
print(aaaa)
