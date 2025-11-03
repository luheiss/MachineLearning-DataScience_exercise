import matplotlib.pyplot as plt
import numpy as np

#config InlineBackend.figure_formats = ["svg"]
np.random.seed(6020)

grad = lambda c, X, y: 2 * X.T @ (X @ c - y)
update = lambda c, delta, X, y: c - delta * grad(c, X, y)


def gd(c, delta, X, y, n, stop=1e-10):
    diff = 1
    for _ in range(1, n):
        cnew = update(c, delta, X, y)
        diff = np.linalg.norm(cnew - c)
        c = cnew
        if diff < stop:
            break
    return c


# The data
x = np.arange(1, 11)
y = np.array([0.2, 0.5, 0.3, 0.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2]).reshape((-1, 1))

X = np.array([x, np.ones(x.shape)]).T
delta = 0.002
c = np.random.random((2, 1))

c_10 = gd(c, delta, X, y, 50)
c_20 = gd(c_10, delta, X, y, 50)
c_30 = gd(c_20, delta, X, y, 200)
p4 = np.linalg.pinv(X) @ y

xf = np.arange(0, 11, 0.1)
y1 = np.polyval(c_10, xf)
y2 = np.polyval(c_20, xf)
y3 = np.polyval(c_30, xf)
y4 = np.polyval(p4, xf)

fig = plt.figure()
plt.plot(x, y, "o", color="r", label="observations")
plt.plot(xf, y1, label=r"$n=50$")
plt.plot(xf, y2, label=r"$n=100$")
plt.plot(xf, y3, label=r"$n=300$")
plt.plot(xf, y4, label=r"$E_2$")
plt.ylim(0, 4)
plt.xlim(0, 11)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper left")
plt.gca().set_aspect(1)
plt.grid(visible=True)
plt.show()