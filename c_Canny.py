import numpy as np
import cv2
from c_NMS import NMS

class Canny(NMS):
    def __init__(self, img):
        super().__init__(img)
        
    def apply_Canny(self, low=20, high=100):
        nms = self.apply_nms()
        strong = nms >= high
        mid = (nms >= low) & (nms < high)

        while True:
            new_mid = np.zeros_like(mid)
            for i in range(1, mid.shape[0] - 1):
                for j in range(1, mid.shape[1] - 1):
                    if mid[i, j]:
                        if np.any(strong[i-1:i+2, j-1:j+2]):
                            new_mid[i, j] = True
            if np.array_equal(new_mid, mid):
                break
            mid = new_mid
        combined = (strong | mid).astype(np.uint8)
        return combined * 255