import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img.jpg", 0)
b = cv2.GaussianBlur(img, (5, 5), 1.4).astype(float)
u = cv2.Sobel(b, cv2.CV_64F, 1, 0)
v = cv2.Sobel(b, cv2.CV_64F, 0, 1)
g = np.hypot(u, v)
d = np.rad2deg(np.arctan2(v, u)) % 180

h, w = g.shape
n = np.zeros((h, w))
for i in range(1, h - 1):
    for j in range(1, w - 1):
        a = d[i, j]
        if a < 22.5 or a >= 157.5:
            p, q = g[i, j - 1], g[i, j + 1]
        elif a < 67.5:
            p, q = g[i - 1, j - 1], g[i + 1, j + 1]
        elif a < 112.5:
            p, q = g[i - 1, j], g[i + 1, j]
        else:
            p, q = g[i - 1, j + 1], g[i + 1, j - 1]
        if g[i, j] >= p and g[i, j] >= q:
            n[i, j] = g[i, j]

low, high = 50, 150
e = np.zeros((h, w), np.uint8)
e[n >= high] = 255
s = list(zip(*np.where(n >= high)))
while s:
    i, j = s.pop()
    for y in range(i - 1, i + 2):
        for x in range(j - 1, j + 2):
            if 0 <= y < h and 0 <= x < w and e[y, x] == 0 and n[y, x] >= low:
                e[y, x] = 255
                s.append((y, x))

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("original")
axes[1].imshow(e, cmap="gray")
axes[1].set_title("mine")
axes[2].imshow(cv2.Canny(img, low, high), cmap="gray")
axes[2].set_title("opencv")
plt.tight_layout()
plt.savefig("2.png")
plt.show()
