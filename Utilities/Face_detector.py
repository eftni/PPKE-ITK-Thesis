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
from keras.utils.conv_utils import convert_kernel
from keras.activations import relu
from keras import backend as K
import numpy as np
import cv2

# input image dimensions
img_rows, img_cols = 128, 128


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    # from tensorflow import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                                    output_names, freeze_var_names)
        return frozen_graph


def channel_pad(tensor, size):
    '''shape = tf.keras.backend.int_shape(tensor)
    print(tensor)
    pad = tf.placeholder(tf.float32,shape=[None, shape[1], shape[2], size])
    return tf.concat([tensor, pad], axis=3)'''
    dims = [[0, 0], [0, 0], [0, 0], [0, size]]
    pad = tf.pad(tensor, tf.convert_to_tensor(dims))
    trans1 = tf.transpose(pad, [0, 3, 2, 1])
    trans2 = tf.transpose(trans1, [0, 3, 2, 1])
    return trans2


def convblock(prev, dfilters, filters):
    r = Activation('relu')(prev)
    conv1 = DepthwiseConv2D(kernel_size=3, activation=None, depth_multiplier=1, strides=1, padding="same",
                            use_bias=False)(r)
    conv2 = Conv2D(filters=filters, kernel_size=1, activation=None, strides=1, padding="valid", use_bias=True)(conv1)
    # a = Add()([r, conv2])
    a = Lambda(lambda x: tf.add(x[0], x[1]))([r, conv2])
    return a


def blazeblock(prev, dfilters, filters, downsample):
    r = Activation('relu')(prev)
    skip = r
    if downsample:
        skip = MaxPool2D(strides=2)(r)
    conv1 = DepthwiseConv2D(kernel_size=3, activation=None, depth_multiplier=1, strides=2 if downsample else 1,
                            padding="same", use_bias=False)(r)
    conv2 = Conv2D(filters=filters, kernel_size=1, activation=None, strides=1, padding="valid", use_bias=True)(conv1)
    diff = filters - dfilters
    pad = Lambda(lambda t: channel_pad(t, diff))(skip)
    # a = Add()([pad, conv2])
    ##trans = tf.transpose(pad, [3,2,1,0])
    a = Lambda(lambda x: tf.add(x[0], x[1]))([pad, conv2])
    return r, a

if K.image_data_format() == 'channels_first':
    input_shape = (1, 3, img_rows, img_cols)
else:
    input_shape = (1, img_rows, img_cols, 3)

input = Input(batch_shape=input_shape)
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
    ##print(len(layer.get_weights()))
    if layer.name.find("depthwise") != -1:
        ker = np.load("weights\\DCONV_" + str(dnum) + ".npy")
        ker = np.moveaxis(ker, 0, 3)
        bias = np.load("weights\\DCONV_" + str(dnum) + "_BIAS.npy")
        dnum = dnum + 1
        layer.set_weights([ker])
        continue

    if layer.name.find("conv2d") != -1:
        ker = np.load("weights\\CONV_" + str(num) + ".npy")
        ker = np.moveaxis(ker, 0, 3)
        bias = np.load("weights\\CONV_" + str(num) + "_BIAS.npy")
        num = num + 1
        layer.set_weights([ker, bias])
        continue

model.save("Test.HD5")

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "model", "Blaze.pb", as_text=False)

cam = cv2.VideoCapture(0)
cv2.namedWindow("Disp", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Disp", (600, 600))
##data = data.astype(np.float32)
run = True
while run:
    ret, pic = cam.read()
    if not ret: continue
    cv2.imshow("Disp", pic)
    pic = np.resize(pic, (128,128,3))
    cl, reg = model.predict(np.expand_dims(pic, axis=0))
    if cv2.waitKey(1) == ord("q"):
        print(cl)
        print(reg)
        run = False


'''model.save("Loaded.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")'''
