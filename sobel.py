import cv2
from c_sobel import sobel
import numpy as np

img = cv2.imread('./homework3/bird.jpg', cv2.IMREAD_GRAYSCALE)

M  = sobel(img).Mag()

disp = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('gradient magnitude', disp)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite('sobel_result.jpg', disp)
cv2.destroyAllWindows()