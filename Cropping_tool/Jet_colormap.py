import cv2
import numpy as np

ROIs = open("..\\Data_saver\\ROIs\\ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]
i = 0

for rect in rects:
    print(i)
    im = cv2.imread("..\\Data_saver\\depth_data\\depth" + str(i) + ".png", cv2.CV_16UC1)
    im = 65535 - im * 42
    im = (im / 256).astype('uint8')
    jet = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    cv2.imshow("Test", jet)
    cv2.waitKey(0);
    ##cv2.imwrite("..\\Data_saver\\depth_color\\col_depth" + str(i) + ".png", jet)
    i = i + 1