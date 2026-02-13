import random
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# List of kernels
def linear_kernel(x, y):
    return np.dot(x, y)

def poly_kernel(x, y):
    return (np.dot(x,y) + 1)**7

def rbf_kernel(x, y):
    sigma = 0.1
    return np.exp(-(np.linalg.norm(x - y)**2)/ (2 * (sigma**2)))

kernel = linear_kernel


class_a = np.concatenate(
    (
        np.random.randn(10, 2) * 0.2 + [1.5, -0.1],
        np.random.randn(10, 2) * 0.2 + [-1.5, -0.1],
    )
)
class_b = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((class_a, class_b))
targets = np.concatenate((np.ones(class_a.shape[0]), -np.ones(class_b.shape[0])))

n = inputs.shape[0]  # Number of rows (samples)

permute = list(range(n))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


p_mat = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        p_mat[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])


def zerofun(alpha):
    return np.dot(alpha, targets)


def objective(alpha):
    return 0.5 * alpha.T @ p_mat @ alpha - np.sum(alpha)


start = np.zeros(n)

XC = {"type": "eq", "fun": zerofun}

C = None
ret = minimize(objective, start, bounds=[(0, C) for _ in range(n)], constraints=XC)

if ret['success']:
    print("Solution found")
else:
    print("Solution could not be found (non lin sep)")
alpha = ret["x"]
sv = [(alpha[i], inputs[i], targets[i]) for i in range(n) if abs(alpha[i] > 10e-5)]


def b_term():
    sum = 0
    for val in sv:
        sum += val[0] * val[2] * kernel(val[1], sv[0][1])
    return sum - sv[0][2]


def ind(x, y, b):
    sum = 0
    for val in sv:
        sum += val[0] * val[2] * kernel([x, y], val[1])
    return sum - b


b = b_term()

plt.figure(figsize=(10, 8))
plt.title("SVM Descision Boundary with Linear Kernel", fontsize=22)

line1 = plt.plot(
    [p[0] for p in class_a],
    [p[1] for p in class_a],
    "r.",
    label="class A"
)

line2 = plt.plot(
    [p[0] for p in class_b],
    [p[1] for p in class_b],
    "b.",
    label="class B"
)

plt.axis("equal")

xlim = 2.5
ylim = 2.5

xgrid = np.linspace(-xlim, xlim)
ygrid = np.linspace(-ylim, ylim)

grid = np.array([[ind(x, y, b) for x in xgrid] for y in ygrid])
plt.contourf(xgrid, ygrid, grid, 100, cmap="RdBu", alpha=0.6)
plt.colorbar(label="Decision Function Value")

ct = plt.contour(
    xgrid,
    ygrid,
    grid,
    (-1.0, 0.0, 1.0),
    colors=("blue", "black", "red"),
    linewidths=(1, 3, 1),
)
plt.clabel(ct, inline=True, fontsize=8)
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)

lines = [Line2D([0], [0], color="black"), Line2D([0], [0], color="red", marker="o", linestyle="None"), Line2D([0], [0], color="blue", marker="o", linestyle="None")]
plt.legend(lines,["Descision boundary", "class A", "class B"], fontsize=16)
plt.tight_layout()
plt.show()

