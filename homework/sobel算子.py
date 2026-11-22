import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

"""手动实现 Sobel 梯度算子，输入为二维灰度图像。"""
def sobel(image):
    image = image.astype(np.float64)

    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    Ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # 边界填充，保持输出大小不变
    padded = np.pad(image, 1, mode="reflect")

    h, w = image.shape
    Gx = np.zeros((h, w), dtype=np.float64)
    Gy = np.zeros((h, w), dtype=np.float64)

    # 将卷积过程转换成 NumPy 切片运算
    for i in range(3):
        for j in range(3):
            region = padded[i:i+h, j:j+w]
            Gx += Kx[i, j] * region
            Gy += Ky[i, j] * region

    magnitude = np.hypot(Gx, Gy) #逐元素求三角形斜边长
    direction = np.arctan2(Gy, Gx) #逐元素求方向

    return Gx, Gy, magnitude, direction

# 以彩色模式读取图片（BGR格式）
image_color = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
if image_color is None:
    raise FileNotFoundError("无法读取 input.jpg")

# 将彩色图片转换成灰度图
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Sobel 算子处理灰度图
Gx, Gy, magnitude, direction = sobel(image_gray)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# BGR转为RGB
image_color=cv2.cvtColor(image_color,cv2.COLOR_BGR2RGB)

#对数尺度配色方案
limit = max(np.max(np.abs(Gx)), np.max(np.abs(Gy)))
limit = max(float(limit), 1.0)

norm_grad = SymLogNorm(
    linthresh=10,   # 可调，小一点会让弱梯度更明显
    linscale=1,
    vmin=-limit,
    vmax=limit,
    base=10
)

# 原图
axes[0, 0].imshow(image_color)
axes[0, 0].set_title("原图")

# 灰度图
axes[0, 1].imshow(image_gray, cmap="gray")
axes[0, 1].set_title("灰度图")

# 水平梯度 Gx
im1 = axes[0, 2].imshow(Gx, cmap="RdBu_r", norm=norm_grad)
axes[0, 2].set_title("水平梯度 Gx")
fig.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

# 竖直梯度 Gy
im2 = axes[1, 0].imshow(Gy, cmap="RdBu_r", norm=norm_grad)
axes[1, 0].set_title("竖直梯度 Gy")
fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 梯度强度
axes[1, 1].imshow(magnitude, cmap="gray")
axes[1, 1].set_title("梯度强度")

# 梯度方向
im3 = axes[1, 2].imshow(direction, cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[1, 2].set_title("梯度方向")
fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()