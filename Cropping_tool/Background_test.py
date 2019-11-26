import cv2
import numpy as np


cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1200, 300)

for i in range(0,448):
    im = cv2.imread("..\\Data_saver\\color\\test_small_col" + str(i) + ".png")
    #im = cv2.blur(im, (3,3))
    depth = cv2.imread("..\\Data_saver\\depth_data\\test_small_depth" + str(i) + ".png", cv2.CV_16UC1)
    depth = cv2.blur(depth, (5,5))
    depth_col = cv2.imread("..\\Data_saver\\depth_color\\test_small_col_depth" + str(i) + ".png")
    smooth = cv2.GaussianBlur(im, (7,7), 5)
    sharp = cv2.addWeighted(im, 2, smooth, -1, 0)

    _, mask1 = cv2.threshold(depth, 0, 1, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(depth, 1000, 1, cv2.THRESH_BINARY_INV)
    mask = mask1*mask2
    mask[mask > 0] = 3
    mask = mask.astype(np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    im_mask, bgdModel, fgdModel = cv2.grabCut(im, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    im_mask = np.where((im_mask == 2) | (im_mask == 0), 0, 1).astype('uint8')
    im = im * im_mask[:, :, np.newaxis]

    sm_mask, bgdModel, fgdModel = cv2.grabCut(smooth, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    sm_mask = np.where((sm_mask == 2) | (sm_mask == 0), 0, 1).astype('uint8')
    smooth = smooth * sm_mask[:, :, np.newaxis]

    sh_mask, bgdModel, fgdModel = cv2.grabCut(sharp, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    sh_mask = np.where((sh_mask == 2) | (sh_mask == 0), 0, 1).astype('uint8')
    sharp = sharp * sh_mask[:, :, np.newaxis]

    final = im * sh_mask[:, :, np.newaxis]

    cv2.imshow("Image", np.hstack((im, smooth, sharp, final)))
    cv2.waitKey(0)
