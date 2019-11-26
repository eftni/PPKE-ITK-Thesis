import cv2
import numpy as np


def normalize(image):
    cv2.threshold(image, 1000, cv2.THRESH_BINARY, cv2.THRESH_TOZERO_INV, image)

    mask = cv2.compare(image, 200, cv2.CMP_GT)
    ##minval, maxval, _, _ = cv2.minMaxLoc(image, mask)
    ##image = image-minval
    image = image - np.average(image)
    thresh = image.astype(np.float32)
    cv2.normalize(thresh, thresh, 0, 1, cv2.NORM_MINMAX)
    #print(thresh)
    return thresh


foam_set = open("..\\Data_saver\\foam_set.data", "w+b")
for i in range(0, 10):
    print(i)
    im = cv2.imread("..\\Data_saver\\depth_data\\small_foam_depth" + str(i) + ".png", cv2.CV_16UC1)
    norm = normalize(im)
    norm.tofile(foam_set)


master = open("..\\Data_saver\\master.txt")
num_train = master.readline()
num_val = master.readline()
num_test = master.readline()
train_set = open("..\\Data_saver\\train_set.data", "w+b")
for i in range(0, int(num_train)):
    print(i)
    im = cv2.imread("..\\Data_saver\\depth_data\\small_depth" + str(i) + ".png", cv2.CV_16UC1)
    norm = normalize(im)
    norm.tofile(train_set)
    #im.tofile(train_set)

target_set = open("..\\Data_saver\\target_set.data", "w+b")
for i in range(0, int(num_train)):
    print(i)
    im = cv2.imread("..\\Data_saver\\color\\small_col" + str(i) + ".png", cv2.CV_8SC1)
    ##lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    ##lab.tofile(target_set)
    im.tofile(target_set)

val_set = open("..\\Data_saver\\val_set.data", "w+b")
for i in range(0, int(num_val)):
    print(i)
    im = cv2.imread("..\\Data_saver\\depth_data\\vali_small_depth" + str(i) + ".png", cv2.CV_16UC1)
    norm = normalize(im)
    norm.tofile(val_set)
    #im.tofile(test_set)

val_target_set = open("..\\Data_saver\\val_target_set.data", "w+b")
for i in range(0, int(num_val)):
    print(i)
    im = cv2.imread("..\\Data_saver\\color\\vali_small_col" + str(i) + ".png", cv2.CV_8SC1)
    ##lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    ##lab.tofile(val_target_set)
    im.tofile(val_target_set)

test_set = open("..\\Data_saver\\test_set.data", "w+b")
for i in range(0, int(num_test)):
    print(i)
    im = cv2.imread("..\\Data_saver\\depth_data\\test_small_depth" + str(i) + ".png", cv2.CV_16UC1)
    norm = normalize(im)
    norm.tofile(test_set)
    #im.tofile(test_set)

test_target_set = open("..\\Data_saver\\test_target_set.data", "w+b")
for i in range(0, int(num_test)):
    print(i)
    im = cv2.imread("..\\Data_saver\\color\\test_small_col" + str(i) + ".png", cv2.CV_8SC1)
    ##lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    ##lab.tofile(test_target_set)
    im.tofile(test_target_set)
