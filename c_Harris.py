import cv2
import numpy as np
from c_sobel import sobel

class Harris(sobel):
    def __init__(self, img, k=0.04):
        super().__init__(img)
        self.k = k

    def Ix2(self):
        return self.Gx() ** 2

    def Iy2(self):
        return self.Gy() ** 2

    def Ixy(self):
        return self.Gx() * self.Gy()

    def get_R(self,sigma=1):
        Ix2 = self.Ix2()
        Iy2 = self.Iy2()
        Ixy = self.Ixy()

        Sx2 = cv2.GaussianBlur(Ix2, (3, 3), sigma)
        Sy2 = cv2.GaussianBlur(Iy2, (3, 3), sigma)
        Sxy = cv2.GaussianBlur(Ixy, (3, 3), sigma)

        det_M = (Sx2 * Sy2) - (Sxy ** 2)
        tr_M = Sx2 + Sy2
        R = det_M - self.k * (tr_M ** 2)

        return R

    def get_harris(self,T=0.01):
        R = self.get_R()
        corners = np.zeros_like(R, dtype=np.uint8)
        corners[R > T] = 1
        Harrised = cv2.normalize(R*corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return Harrised

    def NMS(self, T=0.01):
        h = self.get_harris(T)
        M_padded = np.pad(h, pad_width=1, mode='constant', constant_values=0)
        center = M_padded[1:-1, 1:-1]
        
        left  = M_padded[1:-1, :-2]
        right = M_padded[1:-1, 2:]
        up    = M_padded[:-2, 1:-1]
        down  = M_padded[2:, 1:-1]
        
        up_left     = M_padded[:-2, :-2]
        up_right    = M_padded[:-2, 2:]
        down_left   = M_padded[2:, :-2]
        down_right  = M_padded[2:, 2:]

        keep_mask = (center > left) & (center > right) & (center > up) & (center > down) & \
               (center > up_left) & (center > up_right) & (center > down_left) & (center > down_right)
        

        return keep_mask *255
