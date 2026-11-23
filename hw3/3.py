import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img.jpg")
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
u = cv2.Sobel(g, cv2.CV_64F, 1, 0)
v = cv2.Sobel(g, cv2.CV_64F, 0, 1)
a = cv2.GaussianBlur(u * u, (5, 5), 1)
b = cv2.GaussianBlur(v * v, (5, 5), 1)
c = cv2.GaussianBlur(u * v, (5, 5), 1)
r = a * b - c * c - 0.04 * (a + b) ** 2
m = (r > 0.01 * r.max()) & (r == cv2.dilate(r, None))

out = img.copy()
for y, x in zip(*np.where(m)):
    cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)

plt.figure(figsize=(10, 7))
plt.imshow(out[:, :, ::-1])
plt.title(f"harris corners: {m.sum()}")
plt.tight_layout()
plt.savefig("3.png")
plt.show()
