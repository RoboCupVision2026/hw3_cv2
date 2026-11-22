# 计算机视觉作业三

本目录实现了灰度直方图均衡化、Sobel 边缘检测、非极大值抑制、Canny 边缘检测和 Harris 角点检测。

## 文件说明

### 以 `c_` 前缀命名的核心实现

以下文件主要包含可复用的算法类。

| 文件 | 功能 |
| --- | --- |
| `c_sobel.py` | 实现可分离的 Sobel 平滑与差分卷积，计算梯度幅值、方向和方向扇区 |
| `c_NMS.py` | 根据梯度方向对八邻域梯度幅值进行非极大值抑制 |
| `c_Canny.py` | 在 NMS 结果上进行双阈值处理，并连接与强边缘相邻的中间边缘 |
| `c_Harris.py` | 计算 Harris 矩阵分量、角点响应值，并对角点响应进行非极大值抑制 |

### 其余可运行程序

以下文件主要负责调用核心算法、显示图像或保存结果。

| 文件 | 功能 |
| --- | --- |
| `equal_hist.py` | 使用 NumPy 手动计算直方图和 CDF，并与 OpenCV 的直方图均衡化结果并排显示 |
| `sobel.py` | 调用 `sobel` 类显示 Sobel 梯度幅值 |
| `NMS.py` | 调用 `NMS` 类显示梯度幅值经非极大值抑制后的结果 |
| `Canny` | 调用 `Canny` 类运行边缘检测，使用低阈值 `110` 和高阈值 `320` |
| `Harris.py` | 调用 `Harris` 类显示三个梯度乘积矩阵分量及标记后的角点 |

### 测试素材与结果

- `bird.jpg`：原始测试图像。
- `equalized.jpg`：原图、手动均衡化和 OpenCV 均衡化的横向对比。
- `sobel_result.jpg`：Sobel 梯度幅值。
- `NMSed.jpg`：Harris 角点经非极大值抑制后的结果。
- `Ix2.jpg`、`Iy2.jpg`、`Ixy.jpg`：Harris 算法的三个梯度矩阵分量。
- `Canny_result.jpg`：Canny 边缘检测结果。

## 环境依赖

建议使用 Python 3，并安装：

```bash
pip install numpy opencv-python
```

## 运行方法

当前脚本使用相对工作区根目录的路径读取图片。请从项目根目录运行，例如：

```bash
cd /home/dengyuhan/project/robocup_club/visual_group/learning_task_3

python homework3/equal_hist.py
python homework3/sobel.py
python homework3/NMS.py
python homework3/Canny
python homework3/Harris.py
```

所有可视化脚本都会弹出 OpenCV 窗口。按 `s` 保存当前结果，按任意其他键关闭窗口而不保存。
