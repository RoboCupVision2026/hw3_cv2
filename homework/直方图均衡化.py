import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import cast

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

"""手动实现CDF函数，输入uint8灰度图片，返回长256数字，表示映射关系"""
def CDF(gray: np.ndarray) -> np.ndarray:
    counts = np.bincount(gray.ravel(), minlength=256)  # 统计每个灰度值的像素数
    idx = np.flatnonzero(counts)  # 每个非零元素的下标
    if idx.size == 1:  # 纯色图片，保持原灰度不变
        return np.arange(256,1)
    lower = idx[0]  # 最低灰度值
    counts = np.cumsum(counts) / gray.size  # 每个位置表示亮度低于下标的像素的百分比
    return np.clip(
        np.round((counts - counts[lower]) / (1 - counts[lower]) * 255),
        a_min=0,
        a_max=255,
    ).astype(np.uint8)  # 将原像素亮度映射至0-255

# 以彩色模式读取图片（BGR格式）
image_color = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
if image_color is None:
    raise FileNotFoundError("无法读取 test.jpg")

#BGR转为RGB
image_color=cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# 将彩色图片转换成灰度图 uint8
image_gray = cast(npt.NDArray[np.uint8],cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY))

# 用高级索引完成重映射
mapping = CDF(image_gray)
image_cvt = mapping[image_gray]

#可视化展示
fig, axes = plt.subplots(1, 3, figsize=(14, 9))

axes[0].imshow(image_color)
axes[0].set_title("原图")

axes[1].imshow(image_gray, cmap="gray")
axes[1].set_title("灰度图")

axes[2].imshow(image_cvt, cmap="gray")
axes[2].set_title("转化图")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()