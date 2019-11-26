#######################################
# https://arxiv.org/pdf/1811.03120.pdf
# 6s/epoch on Nvdia GTX 1050
#
#
#
#######################################

from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, BatchNormalization, Input, Concatenate
from keras import backend as K
import numpy as np
from DSSIM import DSSIMObjective as DSSIM

batch_size = 25
epochs = 1000

# input image dimensions
img_rows, img_cols = 128, 128


def downsample_block(prev, filters, size):
    conv1 = Conv2D(filters, size, activation='relu', strides=1, padding="same")(prev)
    conv2 = Conv2D(filters, size, activation='relu', strides=1, padding="same")(conv1)
    pool = AveragePooling2D(pool_size=(2, 2))(conv2)
    out = BatchNormalization()(pool)
    return out

def upsample_block(prev, filters, size):
    convT = Conv2DTranspose(filters, size, activation='relu', strides=2, padding="same")(prev)
    conv1 = Conv2D(filters, size, activation='relu', strides=1, padding="same")(convT)
    conv2 = Conv2D(filters, size, activation='relu', strides=1, padding="same")(conv1)
    out = BatchNormalization()(conv2)
    return out

def output_block(prev, filters, size):
    convT = Conv2DTranspose(filters, size, activation='relu', strides=2, padding="same")(prev)
    conv1 = Conv2D(filters, size, activation='relu', strides=1, padding="same")(convT)
    out = Conv2D(3, size, activation='relu', strides=1, padding="same")(conv1)
    return out

def processor_block(prev, filters, size):
    conv1 = Conv2D(filters, size, activation='relu', strides=1, padding="same")(prev)
    conv2 = Conv2D(filters, size, activation='relu', strides=1, padding="same")(conv1)
    out = BatchNormalization()(conv2)
    return out

# the data, split between train and test sets
train_set = np.fromfile("train_set.data", np.float32)
target_set = np.fromfile("target_set.data", np.uint8)
test_set = np.fromfile("test_set.data", np.float32)
test_target_set = np.fromfile("test_target_set.data", np.uint8)

master = open("master.txt")
num_train = int(master.readline())
num_test = int(master.readline())
print(num_train, num_test)


if K.image_data_format() == 'channels_first':
    train_set = train_set.reshape(num_train, 1, img_rows, img_cols)
    target_set = target_set.reshape(num_train, 3, img_rows, img_cols)
    test_set = test_set.reshape(num_test, 1, img_rows, img_cols)
    test_target_set = test_target_set.reshape(num_test, 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_set = train_set.reshape(num_train, img_rows, img_cols, 1)
    target_set = target_set.reshape(num_train, img_rows, img_cols, 3)
    test_set = test_set.reshape(num_test, img_rows, img_cols, 1)
    test_target_set = test_target_set.reshape(num_test, img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 1)

target_set = target_set/255
test_target_set = test_target_set/255

DSSIM_test = DSSIM()
small = target_set[0:25]
small_other = test_target_set[0:25]
same = DSSIM_test(small, small)
diff = DSSIM_test(small, small_other)
print(same)
print(diff)
exit(1)

print('train_set shape:', train_set.shape)
print('target_set shape:', target_set.shape)
print('train_set shape:', test_set.shape)
print('test_target_set shape:', test_target_set.shape)
print(train_set.shape[0], 'train samples')
print(test_set.shape[0], 'test samples')


input = Input(input_shape)
down1 = downsample_block(input, 16, 5)
proc1 = processor_block(down1, 32, 5)
proc2 = processor_block(proc1, 32, 5)
down2 = downsample_block(down1, 32, 5)
proc3 = processor_block(down2, 32, 5)
down3 = downsample_block(down2, 64, 5)
up1 = upsample_block(down3, 32, 5)
conc1 = Concatenate(axis=3)([proc3, up1])
up2 = upsample_block(conc1, 16, 5)
conc2 = Concatenate(axis=3)([proc2, up2])
out = output_block(conc2, 32, 5)

model = Model(input=input, output=out)

assert model.output_shape == (None, 128, 128, 3)

model.compile(loss=DSSIM(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(train_set, target_set,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
score = model.evaluate(test_set, test_target_set, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

