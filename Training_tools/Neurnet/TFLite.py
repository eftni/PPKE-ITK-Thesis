# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import cv2

import keras
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, BatchNormalization, Input, Concatenate, Dropout, Add
import tensorflow as tf

img_rows, img_cols = 128, 128

if __name__ == '__main__':
    interpreter = tf.lite.Interpreter(model_path='contours.tfl')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    input_shape = input_details[0]['shape']

    cv2.namedWindow("Landmarks", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Landmarks", (600, 600))
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)  # width
    cam.set(4, 720)  # height
    cam.set(10, 255)  # brightness     min: 0   , max: 255 , increment:1
    cam.set(11, 50)  # contrast       min: 0   , max: 255 , increment:1
    cam.set(12, 70)  # saturation     min: 0   , max: 255 , increment:1
    cam.set(13, 13)  # hue         
    cam.set(14, 50)  # gain           min: 0   , max: 127 , increment:1
    cam.set(15, -3)  # exposure       min: -7  , max: -1  , increment:1
    cam.set(17, 5000)  # white_balance  min: 4000, max: 7000, increment:1
    cam.set(28, 0)  # focus          min: 0   , max: 255 , increment:5'''
    run = True
    i = 0
    while run:
        #image = target_set[i]
        ret, image = cam.read()
        if not ret:
            break
        shape = image.shape
        ratioy = shape[0]/192
        ratiox = shape[1]/192
        small = cv2.resize(image.astype(np.float32)/255, (192, 192))
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(small, axis=0))
        interpreter.invoke()
        output1 = interpreter.get_tensor(output_details[0]['index'])
        landmarks = output1[0,0,0]
        output2 = interpreter.get_tensor(output_details[1]['index'])
        j = 0
        final = small
        while j < len(landmarks)-2:
        ##while j < 10:
            ##cv2.circle(image, (int(landmarks[j]*ratiox), int(landmarks[j+1]*ratioy)), 5, (0, 0, 255), -1)
            j += 2
        ##cv2.putText(image, str(i), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Landmarks", image)
        i += 1
        if cv2.waitKey(1) == ord('q'):
            run = False


    '''data = np.zeros(input_shape, np.float32)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    np.save("Lite_reg", results)
    np.savetxt("Lite_reg.txt", results)

    output_data = interpreter.get_tensor(output_details[1]['index'])
    results = np.squeeze(output_data)
    np.save("Lite_class", results)
    np.savetxt("Lite_class.txt", results)

    details = interpreter.get_tensor_details()
    num = 1
    dnum = 1
    for i, det in enumerate(details):
        if det["name"].find("Kernel") != -1:
            ##print(det)
            fname = ""
            if det["name"].find("depthwise") != -1:
                fname = "DCONV_" + str(dnum)
            else:
                fname = "CONV_" + str(num)
            ##print(fname)
            np.save("weights\\" + fname, interpreter.get_tensor(i), allow_pickle=False)
        if det["name"].find("Bias") != -1:
            ##print(det)
            if det["name"].find("depthwise") != -1:
                fname = "DCONV_" + str(dnum)
                dnum = dnum + 1
            else:
                fname = "CONV_" + str(num)
                num = num + 1
            fname += "_BIAS"
            np.save("weights\\" + fname, interpreter.get_tensor(i), allow_pickle=False)'''
