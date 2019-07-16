from scipy import optimize
import numpy as np


def fit_fun(x):
    return 10 + 5 * x


x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = fit_fun(x) + 0.2 * np.random.rand(1, 100)
y1 = y.reshape(100, )


def residuals(p, y, x):  #
    "计算以p为参数的直线和原始数据之间的误差"
    k, b = p
    return (k * x + b) - y


p_init = np.random.randn(2)  # 随机初始化多项式参数
r = optimize.leastsq(residuals, p_init, args=(y1, x))
k, b = r[0]
print("k =", k, "b =", b)
