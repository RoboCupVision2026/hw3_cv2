import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img.jpg", 0)
h = np.bincount(img.ravel(), minlength=256)
c = h.cumsum()
m = c[c > 0].min()
t = np.round((c - m) / (c[-1] - m) * 255).clip(0, 255).astype(np.uint8)
out = t[img]

figure, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0][0].imshow(img, cmap="gray", vmin=0, vmax=255)
axes[0][0].set_title("original")
axes[0][1].imshow(out, cmap="gray", vmin=0, vmax=255)
axes[0][1].set_title("equalized")
axes[1][0].bar(range(256), h)
axes[1][0].set_title("histogram before")
axes[1][1].bar(range(256), np.bincount(out.ravel(), minlength=256))
axes[1][1].set_title("histogram after")
plt.tight_layout()
plt.savefig("4.png")
plt.show()
