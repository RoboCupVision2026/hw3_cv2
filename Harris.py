from c_Harris import Harris
import cv2
import numpy as np

img = cv2.imread('./homework3/bird.jpg', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('./homework3/bird.jpg')

Ix2 = cv2.normalize(Harris(img).Ix2(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Iy2 = cv2.normalize(Harris(img).Iy2(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Ixy = cv2.normalize(Harris(img).Ixy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Ix2', Ix2)
cv2.imshow('Iy2', Iy2)
cv2.imshow('Ixy', Ixy)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite('Ix2.jpg', Ix2)
    cv2.imwrite('Iy2.jpg', Iy2)
    cv2.imwrite('Ixy.jpg', Ixy)

NMSed = Harris(img).NMS()
y_coords, x_coords = np.where(NMSed > 0)
for i in range(len(x_coords)):
        x = int(x_coords[i])
        y = int(y_coords[i])
        cv2.circle(img_color, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imshow('NMSed', img_color)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite('NMSed.jpg', img_color)
cv2.destroyAllWindows()