# hw3

| 文件 | 内容 |
|---|---|
| 1.py | 手动实现 Sobel 算子，对图像滤波，结果 1.png |
| 2.py | 手动实现 Canny：高斯模糊 → Sobel 梯度 → 非极大值抑制 → 双阈值 + 8 连通边缘连接，与 cv2.Canny 对比，结果 2.png |
| 3.py | 手动实现 Harris：梯度 → 结构张量 → 响应 R = det - k·trace² → 阈值 + 局部极大，结果 3.png |
| 4.py | 手动实现直方图均衡化：直方图 → 累积分布 → 映射表，结果 4.png |
| img.jpg | 测试图片 |

运行：`python3 1.py`（依赖 numpy、opencv-python、matplotlib）
