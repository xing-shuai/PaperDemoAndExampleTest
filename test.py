import numpy as np

# beta = 12
# y_est = np.array([[1.1, 3.0, 1.1, 1.3, 0.8]])
# a = np.exp(beta * y_est)
# b = np.sum(np.exp(beta * y_est))
# softmax = a / b
# max = np.sum(softmax * y_est)
# print(max)
# pos = range(y_est.size)
# print(pos)
# softargmax = np.sum(softmax * pos)
# print(softargmax)

a = np.random.randn(100, 100) * 50
a = a + -np.min(a)

a = np.exp(a) / np.sum(np.exp(a))
# print(np.sum(a))
sum_r = np.sum(a, axis=1)
sum_c = np.sum(a, axis=0)

# print(a)

# print(sum_c)

# print(sum_c * np.arange(0, sum_c.shape[0]))

print(np.sum(sum_c * np.arange(0, sum_c.shape[0])))

print(np.argmax(a))

print(np.unravel_index(np.argmax(a), a.shape))

print(np.sum(np.reshape(a, (a.shape[0] * a.shape[1],)) * np.arange(0, a.shape[0] * a.shape[1])))
