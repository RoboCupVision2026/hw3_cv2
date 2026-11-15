
hw3：图像处理算法实现

1. 项目简介

本项目使用 Python、NumPy 和 OpenCV 实现常见的图像处理算法，包括：

1. Sobel 梯度算子
2. 手动 Canny 边缘检测,并与cv2.Canny（）对比
3. 手动 Harris 角点检测
4. 手动直方图均衡化

2. 开发环境

* Python 3.12
* OpenCV
* NumPy
* Matplotlib

安装依赖：

pip install opencv-python numpy matplotlib

3. 项目结构

hw3_cv2/
├── pig.jpg
└── 朱宇涵_cv2_hw3.ipynb

4. 算法说明

4.1 Sobel 梯度算子

利用水平和垂直方向的 Sobel 卷积核计算图像梯度，并计算梯度幅值和方向。

4.2 Canny 边缘检测

按照以下步骤实现：

1. 高斯滤波
2. Sobel 梯度计算
3. 非极大值抑制
4. 双阈值检测
5. 边缘连接

4.3 Harris 角点检测

按照以下步骤实现：

1. 图像梯度计算
2. 梯度乘积计算
3. 高斯滤波
4. 构造二阶矩矩阵
5. 计算 Harris 响应值
6. 阈值筛选
7. 非极大值抑制

4.4 直方图均衡化

通过统计图像灰度直方图、计算累积分布函数和灰度映射，实现图像对比度增强。
