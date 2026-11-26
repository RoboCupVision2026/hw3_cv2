import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei','WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus']=False

KX=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64)
KY=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float64)

def gaussian_kernel(ksize,sigma):
    k=np.zeros((ksize,ksize))
    c=ksize//2
    for i in range(ksize):
        for j in range(ksize):
            k[i][j]=np.exp(-((i-c)**2+(j-c)**2)/(2*sigma**2))
    return k/k.sum()

def conv2d(img,kernel):
    h,w=img.shape
    kh,kw=kernel.shape
    ph,pw=kh//2,kw//2
    padded=np.zeros((h+2*ph,w+2*pw))
    padded[ph:ph+h,pw:pw+w]=img
    # 边缘复制，省得边界黑一圈
    padded[:ph,:]=padded[ph:ph+1,:]
    padded[-ph:,:]=padded[-ph-1:-ph,:]
    padded[:,:pw]=padded[:,pw:pw+1]
    padded[:,-pw:]=padded[:,-pw-1:-pw]

    out=np.zeros((h,w))
    for i in range(kh):
        for j in range(kw):
            out += kernel[i][j]*padded[i:i+h,j:j+w]
    return out

def canny(img,low,high,ksize=5,sigma=1.4):
    if img.ndim==3:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray=img.astype(np.float64)

    blur=conv2d(gray,gaussian_kernel(ksize,sigma))
    gx=conv2d(blur,KX)
    gy=conv2d(blur,KY)

    mag=np.hypot(gx,gy)
    mag=mag/mag.max()*255
    ang=np.arctan2(gy,gx)*180/np.pi
    ang[ang<0] += 180

    # 非极大值抑制
    h,w=mag.shape
    nms=np.zeros((h,w))
    for i in range(1,h-1):
        for j in range(1,w-1):
            a=ang[i,j]
            m=mag[i,j]
            if a<22.5 or a>=157.5:
                n1,n2=mag[i,j-1],mag[i,j+1]
            elif a<67.5:
                n1,n2=mag[i-1,j+1],mag[i+1,j-1]
            elif a<112.5:
                n1,n2=mag[i-1,j],mag[i+1,j]
            else:
                n1,n2=mag[i-1,j-1],mag[i+1,j+1]
            if m>=n1 and m>=n2:
                nms[i,j]=m

    strong=nms>=high
    weak=(nms>=low)&~strong

    edges=np.zeros((h,w),dtype=np.uint8)
    edges[strong]=255
    # 弱边缘只有连到强边缘才留下，用栈往外扩
    stack=[(y,x) for y,x in zip(*np.where(strong))]
    while stack:
        y,x=stack.pop()
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                ny,nx=y+dy,x+dx
                if 0<=ny<h and 0<=nx<w and weak[ny,nx] and edges[ny,nx]==0:
                    edges[ny,nx]=255
                    stack.append((ny,nx))

    return edges,mag,nms

img=cv2.imread('images/lena.jpg')
low,high=10,25
edges,mag,nms=canny(img,low,high)
# cv2 的阈值是按没归一化的梯度幅值算的，所以给它调大点
cv_edges=cv2.Canny(img,50,120,apertureSize=3,L2gradient=True)
print('阈值',low,high)
print('自己写的边缘点数',np.count_nonzero(edges))
print('cv2.Canny 边缘点数',np.count_nonzero(cv_edges))

overlay=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy()
overlay[edges>0]=[255,0,0]

fig,axes=plt.subplots(2,3,figsize=(13,9))
axes[0][0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
axes[0][0].set_title('原图')
axes[0][1].imshow(mag,cmap='gray')
axes[0][1].set_title('梯度幅值')
axes[0][2].imshow(nms,cmap='gray')
axes[0][2].set_title('非极大值抑制')
axes[1][0].imshow(edges,cmap='gray')
axes[1][0].set_title('双阈值+边缘连接 (low=%d, high=%d)'%(low,high))
axes[1][1].imshow(cv_edges,cmap='gray')
axes[1][1].set_title('cv2.Canny 结果')
axes[1][2].imshow(overlay)
axes[1][2].set_title('叠加在原图上')
for row in axes:
    for ax in row:
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('results/canny.png',dpi=130)
plt.show()
