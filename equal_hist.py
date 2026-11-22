import cv2
import numpy as np

def my_equalize_hist(img_gray):

    hist = np.bincount(img_gray.flatten(), minlength=256)
    cdf = np.cumsum(hist)
    total_pixels = max(1, cdf[-1])
    cdf_normalized = cdf * (255.0 / total_pixels)
    cdf_mapped = np.round(cdf_normalized).astype(np.uint8)
    img_equalized = cdf_mapped[img_gray]
    
    return img_equalized


# 直接读成单通道灰度图
img_gray = cv2.imread("./homework3/bird.jpg", cv2.IMREAD_GRAYSCALE)
my_gray_eq = my_equalize_hist(img_gray)
cv2.imshow('Gray Comparison (Original | Mine | OpenCV)', np.hstack((img_gray, my_gray_eq, cv2.equalizeHist(img_gray))))
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite("equalized.jpg", np.hstack((img_gray, my_gray_eq, cv2.equalizeHist(img_gray))))
cv2.destroyAllWindows()