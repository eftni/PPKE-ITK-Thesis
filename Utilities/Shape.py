import cv2
import numpy as np
import math

shape_dd = 10
norm_dd = 4

##Single pass XY adaptive search: Basically a maximum search: Set the X and Y values to the current pixel, then move one. If the
##Value is the same, the "planar counter" gets incremented. When the value changes, every point in the planar counter gets the new value set as it's neighbour.
##Backwards, every pixel gets the value of the planar counter.

def normals_rect(image, rect):
    params = [int(param) for param in rect.split(' ')]
    w = params[3]+2*shape_dd
    h = params[2]+2*shape_dd
    start_x = params[1] - shape_dd if params[1] - shape_dd > 0 else 0
    start_y = params[0] - shape_dd if params[0] - shape_dd > 0 else 0
    norm_x = np.zeros((w, h), np.float32)
    norm_y = np.zeros((w, h), np.float32)
    for y in range(0, h):
        for x in range(0, w):
            dx = 0
            dy = 0
            if norm_dd <= start_x+x < im.shape[0]-norm_dd and 0 <= start_y + y < im.shape[1]:
                dx = (float(image[start_x+x+norm_dd, start_y+y]) - float(image[start_x+x-norm_dd, start_y+y]))/2.0 ##20.0?
            if norm_dd <= start_y+y < im.shape[1]-norm_dd and 0 <= start_x + x < im.shape[0]:
                dy = (float(image[start_x+x, start_y+y+norm_dd]) - float(image[start_x+x, start_y+y-norm_dd]))/2.0
            len = math.sqrt(dx**2 + dy**2 + 0.1**2)
            norm_x[x, y] = dx/len
            norm_y[x, y] = dy/len
    return (norm_x, norm_y)


def normals(image):
    norm_x = np.zeros((image.shape[0], image.shape[1]), np.float32)
    norm_y = np.zeros((image.shape[0], image.shape[1]), np.float32)
    print(image.shape)
    for y in range(0, image.shape[0]-1):
        for x in range(0, image.shape[1]-1):
            dx = 0
            dy = 0
            if x-norm_dd >= 0 and x+norm_dd <= im.shape[1]-1:
                dx = (float(image[y, x+norm_dd]) - float(image[y, x-norm_dd]))/2.0 ##20.0?
            if y-norm_dd >= 0 and y+norm_dd <= im.shape[0]-1:
                dy = (float(image[y+norm_dd, x]) - float(image[y-norm_dd, x]))/2.0
            len = math.sqrt(dx**2 + dy**2 + 0.1**2)
            norm_x[y, x] = dx/len
            norm_y[y, x] = dy/len
    return (norm_x, norm_y)


def shape_rect(norm_x, norm_y, image, rect):
    params = [int(param) for param in rect.split(' ')]
    w = params[3]
    h = params[2]
    start_x = params[1]
    start_y = params[0]
    shape = np.zeros((w, h, 3), np.uint8)
    for y in range(0, h):
        for x in range(0, w):
            xx = (norm_x[x+shape_dd, y] - norm_x[x-shape_dd, y])/2.0
            xy = (norm_y[x+shape_dd, y] - norm_y[x-shape_dd, y])/2.0
            yx = (norm_x[x, y+shape_dd] - norm_x[x, y-shape_dd])/2.0
            yy = (norm_y[x, y+shape_dd] - norm_y[x, y-shape_dd])/2.0
            div = math.sqrt(abs((xx-yy)**2 + 4*xy*yx))
            arg = xx+yy/div if div > 0 else 0
            res = -1*(2/math.pi)*math.atan(arg)
            shape[x, y, 2] = 0 if res >= 0 else -1*math.floor(255*res)
            shape[x, y, 1] = 255-math.floor(255*res) if res >= 0 else 255+math.floor(255*res)
            shape[x, y, 0] = math.floor(255*res) if res >= 0 else 0
            if image[start_x+x, start_y+y] > 1000 or image[start_x+x, start_y+y] == 0:
                shape[x, y, 2] = 0
                shape[x, y, 1] = 0
                shape[x, y, 0] = 0
    return shape


def calc_second_d(im, ix, iy, axis_a, axis_b):
    if(axis_a):
        if(axis_b):
            ff = float(im[ix + shape_dd + norm_dd, iy])
            fs = float(im[ix + shape_dd - norm_dd, iy])
            sf = float(im[ix - shape_dd + norm_dd, iy])
            ss = float(im[ix - shape_dd - norm_dd, iy])
        else:
            ff = float(im[ix + shape_dd, iy + norm_dd])
            fs = float(im[ix + shape_dd, iy - norm_dd])
            sf = float(im[ix - shape_dd, iy + norm_dd])
            ss = float(im[ix - shape_dd, iy - norm_dd])
    else:
        if(axis_b):
            ff = float(im[ix + norm_dd, iy + shape_dd])
            fs = float(im[ix - norm_dd, iy + shape_dd])
            sf = float(im[ix + norm_dd, iy - shape_dd])
            ss = float(im[ix - norm_dd, iy - shape_dd])
        else:
            ff = float(im[ix, iy + shape_dd + norm_dd])
            fs = float(im[ix, iy + shape_dd - norm_dd])
            sf = float(im[ix, iy - shape_dd + norm_dd])
            ss = float(im[ix, iy - shape_dd - norm_dd])

    return ((ff-fs)-(sf-ss))/4


def shape_rect_fast(im, rect):
    params = [int(param) for param in rect.split(' ')]
    w = params[3]
    h = params[2]
    start_x = params[1] + shape_dd + norm_dd
    start_y = params[0] + shape_dd + norm_dd
    shape = np.zeros((w, h), np.float32)
    for y in range(0, h):
        for x in range(0, w):
            ix = start_x + x
            iy = start_y + y
            if im[ix, iy] > 1000 or im[ix, iy] == 0:
                shape[x, y] = 2
            else:
                xx = ((float(im[ix + shape_dd + norm_dd, iy]) - float(im[ix + shape_dd - norm_dd, iy])) - (
                            float(im[ix - shape_dd + norm_dd, iy]) - float(im[ix - shape_dd - norm_dd, iy]))) / 4
                xy = ((float(im[ix + shape_dd, iy + norm_dd]) - float(im[ix + shape_dd, iy - norm_dd])) - (
                            float(im[ix - shape_dd, iy + norm_dd]) - float(im[ix - shape_dd, iy - norm_dd]))) / 4
                yx = ((float(im[ix + norm_dd, iy + shape_dd]) - float(im[ix - norm_dd, iy + shape_dd])) - (
                            float(im[ix + norm_dd, iy - shape_dd]) - float(im[ix - norm_dd, iy - shape_dd]))) / 4
                yy = ((float(im[ix, iy + shape_dd + norm_dd]) - float(im[ix, iy + shape_dd - norm_dd])) - (
                            float(im[ix, iy - shape_dd + norm_dd]) - float(im[ix, iy - shape_dd - norm_dd]))) / 4
                div = math.sqrt(abs((xx-yy)**2 + 4*xy*yx))
                arg = xx+yy/div if div != 0 else 0
                #shape[x, y] = -1*(2/math.pi)*math.atan(arg) ## -1->1
                shape[x, y] = 0.5-(1/math.pi)*math.atan(arg) ## 0->1
    return shape


def shape_dynamic(im, rect):
    params = [int(param) for param in rect.split(' ')]
    w = params[3]
    h = params[2]
    start_x = params[1] + shape_dd + norm_dd
    start_y = params[0] + shape_dd + norm_dd
    shape = np.zeros((w, h), np.float32)
    nans = []
    for y in range(0, h):
        for x in range(0, w):
            ix = start_x + x
            iy = start_y + y
            if im[ix, iy] > 1000 or im[ix, iy] == 0:
                shape[x, y] = 0
            else:
                xx = ((float(im[ix + shape_dd + norm_dd, iy]) - float(im[ix + shape_dd - norm_dd, iy])) - (
                            float(im[ix - shape_dd + norm_dd, iy]) - float(im[ix - shape_dd - norm_dd, iy]))) / 4
                xy = ((float(im[ix + shape_dd, iy + norm_dd]) - float(im[ix + shape_dd, iy - norm_dd])) - (
                            float(im[ix - shape_dd, iy + norm_dd]) - float(im[ix - shape_dd, iy - norm_dd]))) / 4
                yx = ((float(im[ix + norm_dd, iy + shape_dd]) - float(im[ix - norm_dd, iy + shape_dd])) - (
                            float(im[ix + norm_dd, iy - shape_dd]) - float(im[ix - norm_dd, iy - shape_dd]))) / 4
                yy = ((float(im[ix, iy + shape_dd + norm_dd]) - float(im[ix, iy + shape_dd - norm_dd])) - (
                            float(im[ix, iy - shape_dd + norm_dd]) - float(im[ix, iy - shape_dd - norm_dd]))) / 4
                div = math.sqrt(abs((xx-yy)**2 + 4*xy*yx))
                if div != 0:
                    arg = xx+yy/div
                else:
                    arg = 0
                    nans.append([x,y])
                arg = xx+yy/div if div != 0 else 0
                shape[x, y] = 0.5-(1/math.pi)*math.atan(arg) ## 0->1
    return shape

def shape_to_im(shape):
    w = shape.shape[0]
    h = shape.shape[1]
    shape_im = np.zeros((w, h, 3), np.uint8)
    shape = shape*2-1
    for y in range(0, h):
        for x in range(0, w):
            if shape[x, y] != 3:
                res = shape[x, y]
                shape_im[x, y, 2] = 0 if res >= 0 else -1 * math.floor(255 * res)
                shape_im[x, y, 1] = 255 - math.floor(255 * res) if res >= 0 else 255 + math.floor(255 * res)
                shape_im[x, y, 0] = math.floor(255 * res) if res >= 0 else 0
    return shape_im


def shape(norm_x, norm_y, image):
    shape = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
    for y in range(0, im.shape[0]-1):
        for x in range(0, im.shape[1]-1):
            xx = (norm_x[y, x+shape_dd] - norm_x[y, x-shape_dd])/2.0 if x >= shape_dd and x < im.shape[1]-shape_dd-1 else 0
            xy = (norm_y[y, x+shape_dd] - norm_y[y, x-shape_dd])/2.0 if x >= shape_dd and x < im.shape[1]-shape_dd-1 else 0
            yx = (norm_x[y+shape_dd, x] - norm_x[y-shape_dd, x])/2.0 if y >= shape_dd and y < im.shape[0]-shape_dd-1 else 0
            yy = (norm_y[y+shape_dd, x] - norm_y[y-shape_dd, x])/2.0 if y >= shape_dd and y < im.shape[0]-shape_dd-1 else 0
            div = math.sqrt(abs((xx-yy)**2 + 4*xy*yx))
            arg = xx+yy/div if div > 0 else 0
            res = -1*(2/math.pi)*math.atan(arg)
            shape[y, x, 2] = 0 if res >= 0 else -1*math.floor(255*res)
            shape[y, x, 1] = 255-math.floor(255*res) if res >= 0 else 255+math.floor(255*res)
            shape[y, x, 0] = math.floor(255*res) if res >= 0 else 0
            if image[y,x] > 1000 or image[y,x] == 0:
                shape[y, x, 2] = 0
                shape[y, x, 1] = 0
                shape[y, x, 0] = 0
    return shape



master = open("..\\Data_saver\\master.txt")
num_train = master.readline()
num_val = master.readline()
num_test = master.readline()
ROIs = open("..\\Data_saver\\ROIs\\ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]

train_set = open("..\\Data_saver\\Shape_train_set.data", "w+b")
for i in range(0, int(num_train)):
    print(i)
    im = cv2.imread("..\\Data_saver\\depth_data\\depth" + str(i) + ".png", cv2.CV_16UC1)
    im = np.pad(im, (shape_dd+norm_dd, shape_dd+norm_dd), mode='constant', constant_values=0)
    sh2 = shape_rect_fast(im, rects[i])
    sh2 = cv2.resize(sh2, (128, 128))
    sh2.tofile(train_set)
    shim = shape_to_im(sh2)
    cv2.imwrite("..\\Data_saver\\shape_index\\bad_shape" + str(i) + ".png", shim)

ROIs = open("..\\Data_saver\\ROIs\\val_ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]
val_set = open("..\\Data_saver\\Shape_val_set.data", "w+b")
for i in range(0, int(num_val)):
    print(i)
    im = cv2.imread("..\\Data_saver\\depth_data\\vali_depth" + str(i) + ".png", cv2.CV_16UC1)
    im = np.pad(im, (shape_dd+norm_dd, shape_dd+norm_dd), mode='constant', constant_values=0)
    sh2 = shape_rect_fast(im, rects[i])
    sh2 = cv2.resize(sh2, (128, 128))
    sh2.tofile(val_set)
    shim = shape_to_im(sh2)
    cv2.imwrite("..\\Data_saver\\shape_index\\val_shape" + str(i) + ".png", shim)

ROIs = open("..\\Data_saver\\ROIs\\test_ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]
test_set = open("..\\Data_saver\\Shape_test_set.data", "w+b")
for i in range(0, int(num_test)):
    print(i)
    im = cv2.imread("..\\Data_saver\\depth_data\\test_depth" + str(i) + ".png", cv2.CV_16UC1)
    im = np.pad(im, (shape_dd+norm_dd, shape_dd+norm_dd), mode='constant', constant_values=0)
    sh2 = shape_rect_fast(im, rects[i])
    sh2 = cv2.resize(sh2, (128, 128))
    sh2.tofile(test_set)
    shim = shape_to_im(sh2)
    cv2.imwrite("..\\Data_saver\\shape_index\\test_shape" + str(i) + ".png", shim)
