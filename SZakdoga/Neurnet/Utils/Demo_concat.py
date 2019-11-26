import cv2
import numpy as np

demo_name = "Shapelab"

final = np.zeros((1,1))
for num in range(0,24):
    tmp = cv2.imread("C:\\Users\\Niko\\Desktop\\Demo\\" + demo_name + str(num) + ".png")
    if num == 0:
        final = tmp
    else:
        final = np.hstack((final, tmp))
cv2.imwrite("C:\\Users\\Niko\\Desktop\\Demo\\" + demo_name + "_final.png", final)