from c_NMS import NMS
import cv2
import numpy as np

img = cv2.imread('./homework3/bird.jpg', cv2.IMREAD_GRAYSCALE)

NMSed = NMS(img).apply_nms()

disp = cv2.normalize(NMSed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('NMSed', disp)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite('NMS_result.jpg', disp)
cv2.destroyAllWindows()