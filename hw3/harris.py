import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei','WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus']=False

img=cv2.imread('images/lena.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)

KX=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
KY=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32)

ix=cv2.filter2D(gray,-1,KX)
iy=cv2.filter2D(gray,-1,KY)

# 窗口里梯度平方的累加，用高斯加权代替方框
win=(5,5)
ix2=cv2.GaussianBlur(ix*ix,win,1.5)
iy2=cv2.GaussianBlur(iy*iy,win,1.5)
ixy=cv2.GaussianBlur(ix*iy,win,1.5)

k=0.04
R=ix2*iy2-ixy**2-k*(ix2+iy2)**2
print('R 的范围',R.min(),R.max())

# 取局部极大，再卡一个阈值
th=0.01*R.max()
dil=cv2.dilate(R,np.ones((5,5),np.float32))
corners=(R==dil)&(R>th)
ys,xs=np.where(corners)
print('检测到角点数',len(xs))

marked=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy()
for y,x in zip(ys,xs):
    cv2.circle(marked,(x,y),4,(255,0,0),1)

R_cv=cv2.cornerHarris(gray,2,3,0.04)
# cv2 那边没有做非极大值抑制，直接阈值会出来一坨，这里补上再比
dil2=cv2.dilate(R_cv,np.ones((5,5),np.float32))
ys2,xs2=np.where((R_cv==dil2)&(R_cv>0.01*R_cv.max()))
print('cv2.cornerHarris 检测到',len(xs2))
marked2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy()
for y,x in zip(ys2,xs2):
    cv2.circle(marked2,(x,y),4,(0,255,0),1)

fig,axes=plt.subplots(1,3,figsize=(14,5))
axes[0].imshow(marked)
axes[0].set_title('手动实现 Harris (%d 个角点)'%len(xs))
axes[1].imshow(np.log1p(np.clip(R,0,None)),cmap='jet')
axes[1].set_title('角点响应 R')
axes[2].imshow(marked2)
axes[2].set_title('cv2.cornerHarris (%d 个)'%len(xs2))
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('results/harris.png',dpi=130)
plt.show()
