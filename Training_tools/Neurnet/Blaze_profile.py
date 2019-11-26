#######################################
# https://arxiv.org/pdf/1811.03120.pdf
# 6s/epoch on Nvdia GTX 1050
#
#
#
#######################################

from __future__ import print_function
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, DepthwiseConv2D, Add, Concatenate, MaxPool2D, Input, Reshape, Lambda, Activation
from keras.activations import relu
from keras import backend as K
import numpy as np

# input image dimensions
img_rows, img_cols = 128, 128

def channel_pad(tensor, size):
    '''shape = tf.keras.backend.int_shape(tensor)
    print(tensor)
    pad = tf.placeholder(tf.float32,shape=[None, shape[1], shape[2], size])
    return tf.concat([tensor, pad], axis=3)'''
    dims = [[0,0], [0,0], [0,0], [0, size]]
    return tf.pad(tensor, tf.convert_to_tensor(dims))


def convblock(prev, dfilters, filters):
    r = Activation('relu')(prev)
    conv1 = DepthwiseConv2D(kernel_size=3, activation=None, depth_multiplier=1, strides=1, padding="same", use_bias=False)(r)
    conv2 = Conv2D(filters=filters, kernel_size=1, activation=None, strides=1, padding="valid", use_bias=True)(conv1)
    a = Add()([r, conv2])
    return a


def blazeblock(prev, dfilters, filters, downsample):
    r = Activation('relu')(prev)
    skip = r
    if downsample:
        skip = MaxPool2D(strides=2)(r)
    conv1 = DepthwiseConv2D(kernel_size=3, activation=None, depth_multiplier=1, strides=2 if downsample else 1, padding="same", use_bias=False)(r)
    conv2 = Conv2D(filters=filters, kernel_size=1, activation=None, strides=1, padding="valid", use_bias=True)(conv1)
    diff = filters - dfilters
    pad = Lambda(lambda t: channel_pad(t, diff))(skip)
    a = Add()([pad, conv2])
    return r, a




if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

input = Input(input_shape)
conv1 = Conv2D(24, 5, activation=None, strides=2, padding='same')(input)
b1 = convblock(conv1, 24, 24)
_, b2 = blazeblock(b1, 24, 28, False)
_, b3 = blazeblock(b2, 28, 32, True)
_, b4 = blazeblock(b3, 32, 36, False)
_, b5 = blazeblock(b4, 36, 42, False)
_, b6 = blazeblock(b5, 42, 48, True)
_, b7 = blazeblock(b6, 48, 56, False)
_, b8 = blazeblock(b7, 56, 64, False)
_, b9 = blazeblock(b8, 64, 72, False)
_, b10 = blazeblock(b9, 72, 80, False)
_, b11 = blazeblock(b10, 80, 88, False)
skip, b12 = blazeblock(b11, 88, 96, True)
b13 = convblock(b12, 96, 96)
b14 = convblock(b13, 96, 96)
b15 = convblock(b14, 96, 96)
b16 = convblock(b15, 96, 96)
r = Activation('relu')(b16)
out1 = Conv2D(2, 1, activation=None, strides=1, padding='same')(skip)
out2 = Conv2D(6, 1, activation=None, strides=1, padding='same')(r)
out3 = Conv2D(32, 1, activation=None, strides=1, padding='same')(skip)
out4 = Conv2D(96, 1, activation=None, strides=1, padding='same')(r)
reshape1 = Reshape((1, 512, 1))(out1)
reshape2 = Reshape((1, 384, 1))(out2)
reshape3 = Reshape((1, 512, 16))(out3)
reshape4 = Reshape((1, 384, 16))(out4)
classificators = Concatenate(axis=2)([reshape1, reshape2])
regressors = Concatenate(axis=2)([reshape3, reshape4])


model = Model(input=input, output=[classificators, regressors])

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

num = 1
dnum = 1
for i, layer in enumerate(model.layers):
    ##print(layer.name)
    ##print(len(layer.get_weights()))
    if layer.name.find("depthwise") != -1:
        ker = np.load("weights\\DCONV_" + str(dnum) + ".npy")
        ker = np.moveaxis(ker, 0, 3)
        bias = np.load("weights\\DCONV_" + str(dnum) + "_BIAS.npy")
        dnum = dnum+1
        layer.set_weights([ker])
        continue

    if layer.name.find("conv2d") != -1:
        ker = np.load("weights\\CONV_" + str(num) + ".npy")
        ker = np.moveaxis(ker, 0, 3)
        bias = np.load("weights\\CONV_" + str(num) + "_BIAS.npy")
        num = num+1
        layer.set_weights([ker, bias])
        continue

master = open("master.txt")
num_train = int(master.readline())
data = np.fromfile("target_set.data", np.uint8)
data = data.reshape((num_train, 128, 128, 3))
##data = data.astype(np.float32)
for i in range(num_train):
    print(i)
    pic = np.expand_dims(data[i], axis=0)
    cl, reg = model.predict(pic)
    results = np.squeeze(cl)
    np.save("Face_class", results)
    np.savetxt("Face_class.txt", results)
    results = np.squeeze(reg)
    np.save("Lite_reg", results)
    np.savetxt("Lite_reg.txt", results)

model.save("Loaded.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

