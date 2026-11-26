import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei','WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus']=False

sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
sobel_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32)

img=cv2.imread('images/lena.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)

gx=cv2.filter2D(gray,-1,sobel_x)
gy=cv2.filter2D(gray,-1,sobel_y)
mag=np.sqrt(gx**2+gy**2)
print('梯度幅值范围',mag.min(),mag.max())

# 显示的时候取个绝对值，负梯度也是边缘
fig,axes=plt.subplots(1,4,figsize=(16,4.5))
axes[0].imshow(gray,cmap='gray')
axes[0].set_title('灰度图')
axes[1].imshow(np.abs(gx),cmap='gray')
axes[1].set_title('x 方向梯度')
axes[2].imshow(np.abs(gy),cmap='gray')
axes[2].set_title('y 方向梯度')
axes[3].imshow(mag,cmap='gray')
axes[3].set_title('梯度幅值')
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('results/sobel.png',dpi=140)
plt.show()
