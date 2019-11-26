import cv2
import numpy as np


def crop(image, x, y, w, h):
    im_small = image[x:x + w, y:y + h]
    return cv2.resize(im_small, (128, 128))

def strip_BG(col, dep):
    blur = cv2.blur(dep, (5, 5))
    _, mask1 = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(blur, 1000, 1, cv2.THRESH_BINARY_INV)
    mask = mask1*mask2
    mask[mask > 0] = 3
    mask = mask.astype(np.uint8)
    smooth = cv2.GaussianBlur(col, (7,7), 5)
    sharp = cv2.addWeighted(col, 2, smooth, -1, 0)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    zeros = cv2.countNonZero(mask)
    if zeros < 1 or zeros == 128*128:
        return col
    mask, bgdModel, fgdModel = cv2.grabCut(sharp, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return col * mask[:, :, np.newaxis]


ROIs = open("..\\Data_saver\\ROIs\\foam_ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]
print("Cropping foam set")
start = input()
i = int(start)
while i < len(rects):
    print(i)
    params = [int(param) for param in rects[i].split(' ')]
    im = cv2.imread("..\\Data_saver\\color\\foam_col" + str(i) + ".png")
    d_im = cv2.imread("..\\Data_saver\\depth_color\\foam_col_depth" + str(i) + ".png")
    depth = cv2.imread("..\\Data_saver\\depth_data\\foam_depth" + str(i) + ".png", cv2.CV_16UC1)
    im_small = crop(im, params[1], params[0], params[3], params[2])
    d_im_small = crop(d_im, params[1], params[0], params[3], params[2])
    depth_small = crop(depth, params[1], params[0], params[3], params[2])
    im_small = strip_BG(im_small, depth_small)
    cv2.imwrite("..\\Data_saver\\color\\small_foam_col" + str(i) + ".png", im_small)
    cv2.imwrite("..\\Data_saver\\depth_color\\small_foam_col_depth" + str(i) + ".png", d_im_small)
    cv2.imwrite("..\\Data_saver\\depth_data\\small_foam_depth" + str(i) + ".png", depth_small)
    i = i + 1



ROIs = open("..\\Data_saver\\ROIs\\ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]
print("Cropping training set")
start = input()
i = int(start)
while i < len(rects):
    print(i)
    params = [int(param) for param in rects[i].split(' ')]
    im = cv2.imread("..\\Data_saver\\color\\col" + str(i) + ".png")
    d_im = cv2.imread("..\\Data_saver\\depth_color\\col_depth" + str(i) + ".png")
    depth = cv2.imread("..\\Data_saver\\depth_data\\depth" + str(i) + ".png", cv2.CV_16UC1)
    im_small = crop(im, params[1], params[0], params[3], params[2])
    d_im_small = crop(d_im, params[1], params[0], params[3], params[2])
    depth_small = crop(depth, params[1], params[0], params[3], params[2])
    im_small = strip_BG(im_small, depth_small)
    cv2.imwrite("..\\Data_saver\\color\\small_col" + str(i) + ".png", im_small)
    cv2.imwrite("..\\Data_saver\\depth_color\\small_col_depth" + str(i) + ".png", d_im_small)
    cv2.imwrite("..\\Data_saver\\depth_data\\small_depth" + str(i) + ".png", depth_small)
    i = i + 1

ROIs = open("..\\Data_saver\\ROIs\\val_ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]
print("Cropping val set")
start = input()
i = int(start)
while i < len(rects):
    print(i)
    params = [int(param) for param in rects[i].split(' ')]
    im = cv2.imread("..\\Data_saver\\color\\vali_col" + str(i) + ".png")
    d_im = cv2.imread("..\\Data_saver\\depth_color\\vali_col_depth" + str(i) + ".png")
    depth = cv2.imread("..\\Data_saver\\depth_data\\vali_depth" + str(i) + ".png", cv2.CV_16UC1)
    im_small = crop(im, params[1], params[0], params[3], params[2])
    d_im_small = crop(d_im, params[1], params[0], params[3], params[2])
    depth_small = crop(depth, params[1], params[0], params[3], params[2])
    im_small = strip_BG(im_small, depth_small)
    cv2.imwrite("..\\Data_saver\\color\\vali_small_col" + str(i) + ".png", im_small)
    cv2.imwrite("..\\Data_saver\\depth_color\\vali_small_col_depth" + str(i) + ".png", d_im_small)
    cv2.imwrite("..\\Data_saver\\depth_data\\vali_small_depth" + str(i) + ".png", depth_small)
    i = i + 1
ROIs = open("..\\Data_saver\\ROIs\\test_ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]
print("Cropping test set")
start = input()
i = int(start)
while i < len(rects):
    print(i)
    params = [int(param) for param in rects[i].split(' ')]
    im = cv2.imread("..\\Data_saver\\color\\test_col" + str(i) + ".png")
    d_im = cv2.imread("..\\Data_saver\\depth_color\\test_col_depth" + str(i) + ".png")
    depth = cv2.imread("..\\Data_saver\\depth_data\\test_depth" + str(i) + ".png", cv2.CV_16UC1)
    im_small = crop(im, params[1], params[0], params[3], params[2])
    d_im_small = crop(d_im, params[1], params[0], params[3], params[2])
    depth_small = crop(depth, params[1], params[0], params[3], params[2])
    im_small = strip_BG(im_small, depth_small)
    cv2.imwrite("..\\Data_saver\\color\\test_small_col" + str(i) + ".png", im_small)
    cv2.imwrite("..\\Data_saver\\depth_color\\test_small_col_depth" + str(i) + ".png", d_im_small)
    cv2.imwrite("..\\Data_saver\\depth_data\\test_small_depth" + str(i) + ".png", depth_small)
    i = i + 1
