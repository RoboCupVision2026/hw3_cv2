import cv2
import numpy as np
from c_sobel import sobel


class NMS(sobel):
    def __init__(self, img):
        super().__init__(img)

    def apply_nms(self):
        M = self.Mag()
        sectors = self.Sectors()
        M_padded = np.pad(M, pad_width=1, mode='constant', constant_values=0)
        center = M_padded[1:-1, 1:-1]
        
        left  = M_padded[1:-1, :-2]
        right = M_padded[1:-1, 2:]
        up    = M_padded[:-2, 1:-1]
        down  = M_padded[2:, 1:-1]
        
        up_left     = M_padded[:-2, :-2]
        up_right    = M_padded[:-2, 2:]
        down_left   = M_padded[2:, :-2]
        down_right  = M_padded[2:, 2:]

        mask_0 = (sectors == 0) & (center >= left) & (center >= right)
        mask_1 = (sectors == 1) & (center >= up_right) & (center >= down_left)
        mask_2 = (sectors == 2) & (center >= up) & (center >= down)
        mask_3 = (sectors == 3) & (center >= up_left) & (center >= down_right)

        keep_mask = mask_0 | mask_1 | mask_2 | mask_3

        nms_result = np.zeros_like(M)
        nms_result[keep_mask] = center[keep_mask]
        
        return nms_result