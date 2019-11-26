from __future__ import absolute_import
from keras import backend as K
import tensorflow as tf
import numpy as np
import dlib
import cv2
import math


def TF_DSSIM(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, 2)
    dssim = (1-ssim)/2
    return K.mean(dssim)

def TF_DSSIM_LAB(y_true, y_pred):
    ssim = tf.image.ssim(y_true[:, :, :, 0], y_pred[:, :, :, 0], 2)
    loss = K.mean((1.0 - ssim) / 2.0) + K.mean(K.abs((y_true[:, :, :, 1] - y_pred[:, :, :, 1]))) + K.mean(
        K.abs((y_true[:, :, :, 2] - y_pred[:, :, :, 2])))
    return loss

class DSSIMObjective:
    def __init__(self, k1=0.01, k2=0.03, kernel_x=3, kernel_y=3, max_value=1.0):
        self.__name__ = 'DSSIMObjective'
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, y_true, y_pred):
        kernel = [1, self.kernel_x, self.kernel_y, 1]
        true_L = K.reshape(y_true[:,:,:,0], [-1] + list(self.__int_shape(y_pred)[1:3]) + [1])
        pred_L = K.reshape(y_pred[:,:,:,0], [-1] + list(self.__int_shape(y_pred)[1:3]) + [1])

        patches_pred = tf.image.extract_patches(images=pred_L, sizes=kernel, strides=kernel, rates=[1,1,1,1],
                                          padding='VALID')
        patches_true = tf.image.extract_patches(images=true_L, sizes=kernel, strides=kernel, rates=[1,1,1,1],
                                          padding='VALID')

        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = ((K.square(u_true)
                  + K.square(u_pred)
                  + self.c1) * (var_pred + var_true + self.c2))
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        loss = K.mean((1.0 - ssim) / 2.0) + K.mean(K.abs((y_true[:,:,:,1]-y_pred[:,:,:,1]))) + K.mean(K.abs((y_true[:,:,:,2]-y_pred[:,:,:,2])))
        return loss
        #return np.mean(ssim)

class LandmarkObjective:
    def __init__(self, batch):
        self.__name__ = 'LandmarkObjective'
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.rect = dlib.rectangle(0, 0, 128, 128)
        self.batch = batch


    @tf.function
    def __predict(self, pic):
        return self.predictor(K.eval(pic), self.rect)

    def __dist(self, shape_true, shape_pred):
        i = 0
        sum_diff = 0.0
        while i < len(shape_true)-1:
            sum_diff += math.sqrt((shape_true[i]-shape_pred[i])*(shape_true[i]-shape_pred[i]) + (shape_true[i+1]-shape_pred[i+1])*(shape_true[i+1]-shape_pred[i+1]))
            i += 2
        return sum_diff

    def __call__(self, y_true, y_pred):
        loss = 0.0
        shape_true = tf.map_fn(self.__predict, y_true)
        shape_pred = tf.map_fn(self.__predict, y_pred)
        for i in range(self.batch):
            loss += self.__dist(shape_true[i], shape_pred[i])
        return loss / self.batch

def test_loss(predictor):
    rect = dlib.rectangle(0,0,128,128)
    def loss(y_true, y_pred):
        loss_total = 0.0
        print(y_true.shape)
        for i in range(y_true.shape[0]):
            shape_true = tf.map_fn(predictor, y_true[i], rect)
            shape_pred = tf.map_fn(predictor, y_pred[i], rect)
            sum_diff = 0.0
            for i in range(68):
                sum_diff += math.sqrt((shape_true.part(i).x - shape_pred.part(i).x) * (shape_true.part(i).x - shape_pred.part(i).x) + (
                                shape_true.part(i).y - shape_pred.part(i).y) * (shape_true.part(i).y - shape_pred.part(i).y))
            loss_total += sum_diff
        return loss_total / y_true.shape[0]
    return loss

