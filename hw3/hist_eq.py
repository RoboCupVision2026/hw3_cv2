import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei','WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus']=False

def equalize(img):
    hist=np.bincount(img.ravel(),minlength=256)
    cdf=np.cumsum(hist)
    cdf_min=cdf[hist>0].min()
    total=img.size
    lut=np.round((cdf-cdf_min)/(total-cdf_min)*255)
    lut=lut.clip(0,255).astype(np.uint8)
    return lut[img],lut

img=cv2.imread('images/lena.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

eq,lut=equalize(gray)
cv_eq=cv2.equalizeHist(gray)
print('和 cv2.equalizeHist 的最大差异',np.abs(eq.astype(int)-cv_eq.astype(int)).max())

fig,axes=plt.subplots(2,3,figsize=(13,8))
axes[0][0].imshow(gray,cmap='gray')
axes[0][0].set_title('原图')
axes[0][1].hist(gray.ravel(),256,range=(0,256))
axes[0][1].set_title('原图直方图')
axes[0][2].plot(lut)
axes[0][2].set_title('灰度映射曲线')
axes[1][0].imshow(eq,cmap='gray')
axes[1][0].set_title('均衡化之后')
axes[1][1].hist(eq.ravel(),256,range=(0,256))
axes[1][1].set_title('均衡化后直方图')
axes[1][2].imshow(cv_eq,cmap='gray')
axes[1][2].set_title('cv2.equalizeHist 结果')

for ax in [axes[0][0],axes[1][0],axes[1][2]]:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('results/hist_eq.png',dpi=130)
plt.show()

# 彩色图就把 Y 通道拉一下
ycc=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
ycc[:,:,0],_=equalize(ycc[:,:,0])
color_eq=cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR)

fig,axes=plt.subplots(1,2,figsize=(11,5))
axes[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
axes[0].set_title('原图')
axes[1].imshow(cv2.cvtColor(color_eq,cv2.COLOR_BGR2RGB))
axes[1].set_title('Y 通道均衡化后')
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('results/hist_eq_color.png',dpi=130)
plt.show()
