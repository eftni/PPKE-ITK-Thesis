from __future__ import absolute_import
from keras import backend as K
import tensorflow as tf

def SSIMMetric(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, 2)
    return K.mean(ssim)


def SSIMVar(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, 2)
    return K.std(ssim)


def PSNRMetric(y_true, y_pred):
    psnr = tf.image.psnr(y_true, y_pred, 2)
    return K.mean(psnr)


def PSNRVar(y_true, y_pred):
    psnr = tf.image.psnr(y_true, y_pred, 2)
    return K.std(psnr)


def MSEMetric(y_true, y_pred):
    mse = tf.losses.MSE(y_true, y_pred)
    return K.mean(mse)


def MSEVar(y_true, y_pred):
    mse = tf.losses.MSE(y_true, y_pred)
    return K.std(mse)


"""
class SSIMMetric:
    def __init__(self):
        self.__name__ = "SSIM"
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, y_true, y_pred):
        tmp = [y_true, y_pred]
        ssims = tf.map_fn(lambda out: print(out[1].shape), tmp)
        #ssims = tf.map_fn(lambda out: compare_ssim(out[0], out[1], multichannel=True), tmp)
        return K.mean(ssims)
"""