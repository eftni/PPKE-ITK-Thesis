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
import tensorflow as tf

if __name__ == '__main__':
    interpreter = tf.lite.Interpreter(model_path='face_detection_front.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    master = open("master.txt")
    num_train = int(master.readline())
    data = np.fromfile("target_set.data", np.uint8)
    data = data.reshape((num_train, 128, 128, 3))
    data = data.astype(np.float32)

    for i in range(num_train):
        pic = np.expand_dims(data[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], pic)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        output_data = interpreter.get_tensor(output_details[1]['index'])
        results = np.squeeze(output_data)