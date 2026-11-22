import cv2
import numpy as np

class sobel:
    def __init__(self,img):
        self.img = img
        self.SMOOTH = [1, 2, 1]
        self.DIFF   = [-1, 0, 1]

    def convolve1d(self,img, kernel, axis):
        k = np.asarray(kernel, dtype=np.float64)
        r = len(k)//2
        pad = [(0,0),(0,0)]; pad[axis] = (r,r)
        p = np.pad(img.astype(np.float64), pad, mode='edge')
        out = np.zeros_like(img, dtype=np.float64)
        n = img.shape[axis]
        for i,c in enumerate(k):
            sl = [slice(None), slice(None)]; sl[axis] = slice(i, i+n)
            out += c * p[tuple(sl)]
        return out

    def Gx(self):
        return self.convolve1d(self.convolve1d(self.img, self.SMOOTH, 0), self.DIFF, 1)

    def Gy(self):
        return self.convolve1d(self.convolve1d(self.img, self.SMOOTH, 1),self.DIFF, 0)

    def Mag(self):    
        gx = self.Gx()
        gy = self.Gy()
        return np.hypot(gx, gy)
    
    def Direction(self):
        gx = self.Gx()
        gy = self.Gy()
        angle = np.arctan2(gy, gx) * (180.0 / np.pi)
        angle = np.abs(angle)
        angle[angle >= 180] -= 180
        return angle
    
    def Sectors(self):
        angle = self.Direction()
        sectors = np.zeros_like(angle, dtype=np.uint8)
        shifted = (angle + 22.5) % 180
        sectors = (shifted // 45).astype(np.uint8)
        
        return sectors

         