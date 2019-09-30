import numpy as np

a = np.array([[1.1, -0.1],
              [-0.1, 1.1]])

b = np.array([[86.6], [103.4]])

u, d, v = np.linalg.svd(a)

# print(u, d, v)

b_ = u.T @ b

y = [b_[i, 0] / val for i, val in enumerate(d)]

x = v.T @ np.expand_dims(np.array(y), 1)

print(x)
