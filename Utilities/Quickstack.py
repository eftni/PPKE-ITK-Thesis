import cv2
import numpy as np

dirs = ["C:\\Users\\Niko\\Desktop\\Final Architectures\\Autoencoder\\",
        "C:\\Users\\Niko\\Desktop\\Final Architectures\\1.0\\",
        "C:\\Users\\Niko\\Desktop\\Final Architectures\\2.0\\SmallNet\\",
        "C:\\Users\\Niko\\Desktop\\Final Architectures\\2.0\\Largenet\\"]
subdirs = ["MSE 250", "LAB 250", "DSSIM 84", "SMSE 250", "SLAB 250", "SDSSIM 84"]

zero = np.zeros((480, 640, 3))
cv2.imwrite("C:\\Users\\Niko\\Desktop\\balck.png", zero)

for d in dirs:
    first = True
    stack = None
    for sd in subdirs:
        if first:
            stack = cv2.imread(d + sd + "\\out.png")
            first = False
        else:
            stack = np.vstack((stack, cv2.imread(d + sd + "\\out.png")))
    cv2.imwrite(d + "Stack.png", stack)
