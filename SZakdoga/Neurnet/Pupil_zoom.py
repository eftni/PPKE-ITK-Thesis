import cv2

img = cv2.imread("capture.png")

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

eyes = img[350:450, 550:900]

cv2.resizeWindow("Test", (eyes.shape[1]*5, eyes.shape[0]*5))

cv2.imshow("Test", eyes*2)
cv2.waitKey(0)