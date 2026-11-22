import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from numba import njit,prange

plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示为方块

# 之前实现的Sobel算子 向下|向右为正方向 去除梯度方向计算
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

    padded = np.pad(image, 1, mode="reflect")
    h, w = image.shape

    Gx = np.zeros((h, w), dtype=np.float64)
    Gy = np.zeros((h, w), dtype=np.float64)

    for i in range(3):
        for j in range(3):
            region = padded[i:i+h, j:j+w]
            Gx += Kx[i, j] * region
            Gy += Ky[i, j] * region

    magnitude = np.hypot(Gx, Gy)

    return Gx, Gy, magnitude

# 沿梯度方向线性插值的NMS 消除除法
@njit(parallel=True)
def nms_fast(Gx, Gy, magnitude):
    h, w = magnitude.shape
    result = np.zeros_like(magnitude)

    #逐元素遍历
    for y in prange(1, h - 1):
        for x in range(1, w - 1):
            m = magnitude[y, x]
            if m==0.0:
                continue

            gx = Gx[y, x]
            gy = Gy[y, x]
            ax = abs(gx)
            ay = abs(gy)            

            #水平方向
            if ax >= ay:
                p, q = ax, ay
                a1 = magnitude[y, x + 1] #右
                a2 = magnitude[y, x - 1] #左

                #左上/右下方向
                if gx * gy >= 0:
                    b1 = magnitude[y + 1, x + 1] #右下
                    b2 = magnitude[y - 1, x - 1] #左上
                #左下/右上方向
                else:
                    b1 = magnitude[y - 1, x + 1] #右上
                    b2 = magnitude[y + 1, x - 1] #左下

            #竖直方向
            else:
                p, q = ay, ax
                a1 = magnitude[y + 1, x] #下
                a2 = magnitude[y - 1, x] #上

                #左上/右下方向
                if gx * gy >= 0:
                    b1 = magnitude[y + 1, x + 1] #右下
                    b2 = magnitude[y - 1, x - 1] #左上
                #左下/右上方向
                else:
                    b1 = magnitude[y + 1, x - 1] #左下              
                    b2 = magnitude[y - 1, x + 1] #右上

            #插值比较，消除除法
            center = p * m
            neighbor1 = (p - q) * a1 + q * b1
            neighbor2 = (p - q) * a2 + q * b2
            if center >= neighbor1 and center >= neighbor2:
                result[y, x] = m

    return result

# 双阈值分类 bool矩阵
def double_threshold(image, low, high):
    strong = image >= high
    weak = (image >= low) & (image < high)

    return strong, weak

# 滞后阈值追踪（8 邻域 BFS）
def hysteresis(strong, weak):
    h, w = strong.shape
    result = strong.copy()

    # 把所有强边缘作为 BFS 起点
    queue = deque(map(tuple, np.argwhere(strong)))

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    while queue:
        y, x = queue.popleft()

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            if not (0 <= ny < h and 0 <= nx < w):
                continue

            # 弱边缘与已确认边缘连通
            if weak[ny, nx] and not result[ny, nx]:
                result[ny, nx] = True
                queue.append((ny, nx))

    return result.astype(np.uint8) * 255

# 完整 Canny 流程
def manual_canny(image, low=40, high=100, sigma=1.4):
    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=sigma)

    # Sobel
    Gx, Gy, magnitude = sobel(blurred)

    # 非极大值抑制
    nms = nms_fast(Gx, Gy, magnitude)

    # 双阈值
    strong, weak = double_threshold(nms, low, high)

    # 滞后阈值追踪
    edges = hysteresis(strong, weak)

    return blurred, magnitude, nms, strong, weak, edges

# 以彩色模式读取图片（BGR格式）
image_color = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
if image_color is None:
    raise FileNotFoundError("无法读取 input.jpg")

# 将彩色图片转换成灰度图
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

blurred, magnitude, nms, strong, weak, edges = manual_canny(
    image_gray,
    low=40,
    high=90,
    sigma=1.4
)

# 可视化各个阶段
images = [
    blurred,
    magnitude,
    nms,
    strong,
    weak,
    edges
]

titles = [
    "高斯模糊",
    "Sobel梯度",
    "非极大值抑制",
    "强边缘",
    "弱边缘",
    "最终Canny边缘"
]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for ax, img, title in zip(axes.flat, images, titles):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
