import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

#手动实现Harris角点检测
def harris_corner(
    gray: np.ndarray,
    k: float = 0.04,
    sigma: float = 1.5,
    threshold: float = 0.01,
    nms_size: int = 7,
):
    """
    参数:
        gray:       灰度图像，shape=(H, W)
        k:          Harris 响应参数
        sigma:      结构张量高斯平滑标准差
        threshold:  相对于最大正响应的阈值
        nms_size:   非极大值抑制窗口大小（奇数）
    返回:
        corners:    bool 类型角点掩码，shape=(H, W)
        response:   Harris 响应图，shape=(H, W)
        Ix, Iy:     水平、垂直梯度
        A, B, C:    结构张量的三个独立分量
    """
    assert nms_size > 0 and nms_size % 2 == 1

    img = gray.astype(np.float32) / 255.0 #归一化[0,1]控制数据尺度

    # Sobel 梯度计算
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    Ky = Kx.T

    Ix = cv2.filter2D(img, cv2.CV_32F, Kx)
    Iy = cv2.filter2D(img, cv2.CV_32F, Ky)

    # 梯度乘积
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 高斯平滑，构造结构张量
    A = cv2.GaussianBlur(Ixx, (0, 0), sigma)
    B = cv2.GaussianBlur(Iyy, (0, 0), sigma)
    C = cv2.GaussianBlur(Ixy, (0, 0), sigma)
    # M=[[A C][C B]]

    # Harris 响应
    det = A * B - C * C
    trace = A + B
    response = det - k * trace * trace

    # 相对阈值
    max_response = np.max(response)
    if max_response <= 0:
        corners = np.zeros_like(response, dtype=bool)
        return corners, response, Ix, Iy, A, B, C

    mask = response > threshold * max_response #布尔矩阵

    # 非极大值抑制
    kernel = np.ones(
        (nms_size, nms_size), dtype=np.uint8
    )
    local_max = cv2.dilate(response, kernel) #将3x3中最大值赋给中心像素

    corners = mask & (response == local_max) #超过阈值&3x3中最大

    return corners, response, Ix, Iy, A, B, C


# 读取图片
img_bgr = cv2.imread("input.jpg")
if img_bgr is None:
    raise FileNotFoundError("找不到 input.jpg")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 执行 Harris
corners, R, Ix, Iy, A, B, C = harris_corner(
    gray,
    k=0.04,
    sigma=1.5,
    threshold=0.01,
    nms_size=5
)

# np.nonzero 返回 (y坐标数组, x坐标数组)
ys, xs = np.nonzero(corners)

#对数尺度配色方案
limit = max(np.max(np.abs(Ix)), np.max(np.abs(Iy)))
limit = max(float(limit), 1.0)

norm_grad = SymLogNorm(
    linthresh=0.1,   # 图像被归一化，使用小数字
    linscale=1,
    vmin=-limit,
    vmax=limit,
    base=10
)

r_limit = float(np.percentile(np.abs(R), 99.5)) #解决响应值正负数量不匹配
r_limit = max(r_limit, 1e-12)

norm_R = SymLogNorm(
    linthresh=max(r_limit * 0.0004, 1e-12),
    linscale=1,
    vmin=-r_limit,
    vmax=r_limit,
    base=10
)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("原图")

im1=axes[0, 1].imshow(Ix, cmap="RdBu_r",norm=norm_grad)
axes[0, 1].set_title("水平梯度")
fig.colorbar(im1,ax=axes[0,1],fraction=0.046, pad=0.04)

im2=axes[0, 2].imshow(Iy, cmap="RdBu_r",norm=norm_grad)
axes[0, 2].set_title("竖直梯度")
fig.colorbar(im2,ax=axes[0,2],fraction=0.046, pad=0.04)

im3=axes[1, 0].imshow(R, cmap="RdBu_r",norm=norm_R)
axes[1, 0].set_title("Harris响应函数返回值")
fig.colorbar(im3,ax=axes[1,0],fraction=0.046, pad=0.04)

axes[1, 1].imshow(corners, cmap="gray")
axes[1, 1].set_title("极大值抑制")
axes[1, 1].scatter(
    xs, ys,
    s=6,
    c="yellow",
    linewidths=0
)

axes[1, 2].imshow(img_rgb)
axes[1, 2].scatter(
    xs, ys,
    s=10,
    c="red",
    marker="o",
    linewidths=0
)
axes[1, 2].set_title(f"探测到的角点: {len(xs)}")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()