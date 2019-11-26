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
    cv2.namedWindow("Landmarks", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Landmarks", (600, 600))
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)  # width
    cam.set(4, 720)  # height
    cam.set(10, 255)  # brightness     min: 0   , max: 255 , increment:1
    cam.set(11, 255)  # contrast       min: 0   , max: 255 , increment:1
    cam.set(12, 255)  # saturation     min: 0   , max: 255 , increment:1
    cam.set(13, 13)  # hue
    cam.set(14, 127)  # gain           min: 0   , max: 127 , increment:1
    cam.set(15, -1)  # exposure       min: -7  , max: -1  , increment:1
    cam.set(17, 7000)  # white_balance  min: 4000, max: 7000, increment:1
    cam.set(28, 0)  # focus          min: 0   , max: 255 , increment:5'''
    run = True
    i = 0
    test = np.ones((1080, 1920, 3))
    cv2.circle(test, (100, 100), 40, (255, 255, 255), -1)
    cv2.circle(test, (1800, 1000), 40, (255, 255, 255), -1)
    cv2.circle(test, (100, 1000), 40, (255, 255, 255), -1)
    cv2.circle(test, (1800, 100), 40, (255, 255, 255), -1)
    while run:
        ret, image = cam.read()
        if not ret:
            break
        cv2.imshow("Landmarks", image)
        i += 1
        if cv2.waitKey(1) == ord('s'):
            cv2.imwrite("capture.png", image)
        if cv2.waitKey(1) == ord('q'):
            run = False
