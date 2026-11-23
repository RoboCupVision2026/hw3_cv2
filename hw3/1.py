import numpy as np
import matplotlib.pyplot as plt


def conv(img, m):
    k = m.shape[0]
    r = k // 2
    h, w = img.shape
    p = np.pad(img, r, mode="edge")
    out = np.zeros((h, w))
    for i in range(k):
        for j in range(k):
            out += m[i, j] * p[i:i + h, j:j + w]
    return out


img = plt.imread("img.jpg").mean(axis=2)
a = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
b = a.T
x = conv(img, a)
y = conv(img, b)
g = np.hypot(x, y)

figure, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0][0].imshow(img, cmap="gray")
axes[0][0].set_title("original")
axes[0][1].imshow(np.abs(x), cmap="gray")
axes[0][1].set_title("sobel x")
axes[1][0].imshow(np.abs(y), cmap="gray")
axes[1][0].set_title("sobel y")
axes[1][1].imshow(g, cmap="gray")
axes[1][1].set_title("magnitude")
plt.tight_layout()
plt.savefig("1.png")
plt.show()
