from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, BatchNormalization
from keras import backend as K
import numpy as np

batch_size = 25
epochs = 500

# input image dimensions
img_rows, img_cols = 128, 128


def add_downsample(model, filters, size):
    model.add(Conv2D(filters, size, activation='relu', strides=1, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2)))

def add_upsample(model, filters, size):
    model.add(Conv2DTranspose(filters, size, activation='relu', strides=2, padding="same"))
    model.add(BatchNormalization())

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
print('train_set shape:', train_set.shape)
print('target_set shape:', target_set.shape)
print('train_set shape:', test_set.shape)
print('test_target_set shape:', test_target_set.shape)
print(train_set.shape[0], 'train samples')
print(test_set.shape[0], 'test samples')


model = Sequential() # (None, 128, 128, 1)
model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=input_shape))
model.add(AveragePooling2D(pool_size=(2, 2)))   #(None, 64, 64, 32)
add_downsample(model, 64, 3)                #(None, 32, 32, 64)
add_downsample(model, 64, 3)                #(None, 16, 16, 64)
add_downsample(model, 128, 3)               #(None, 8, 8, 128)
add_downsample(model, 128, 3)               #(None, 4, 4, 128)
add_downsample(model, 256, 3)               #(None, 2, 2, 256)
add_downsample(model, 256, 3)               #(None, 1, 1, 256)

model.add(Conv2D(256, 3, activation='relu', strides=1, padding="same"))
model.add(Conv2D(256, 3, activation='relu', strides=1, padding="same"))
model.add(Conv2D(256, 3, activation='relu', strides=1, padding="same"))
model.add(Conv2D(256, 3, activation='relu', strides=1, padding="same"))

add_upsample(model, 512, 3)                 #(None, 2, 2, 512)
add_upsample(model, 256, 3)                 #(None, 4, 4, 256)
add_upsample(model, 256, 3)                 #(None, 8, 8, 256)
add_upsample(model, 128, 3)                  #(None, 16, 16, 128)
add_upsample(model, 128, 3)                  #(None, 32, 32, 128)
add_upsample(model, 64, 3)                  #(None, 64, 64, 64)
#add_upsample(model, 32, 3)                  #(None, 128, 128, 64)

model.add(Conv2DTranspose(3, 3, activation='relu', strides=2, padding="same"))
assert model.output_shape == (None, 128, 128, 3)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#model.fit(train_set, target_set,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1)
score = model.evaluate(test_set, test_target_set, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

