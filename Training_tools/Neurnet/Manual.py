from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import cv2
import csv
from Metrics import *
from Losses import LandmarkObjective
from Losses import test_loss
import dlib
import math

##tf.enable_eager_execution()

img_rows, img_cols = 128, 128

demo = False
demo_name = "Dropout"
LAB = True
shape = False
autoplay = False


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

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
    #from tensorflow import convert_variables_to_constants
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
        frozen_graph = tf.v1.convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


json_file = open('autoencoder.json', 'r')
loaded_json = json_file.read()
json_file.close()
model = model_from_json(loaded_json)
model.load_weights("autoencoder.h5")

train_set = np.fromfile("Shape_train_set.data", np.float32) if shape else np.fromfile("train_set.data", np.float32)
target_set = np.fromfile("LAB_target_set.data", np.uint8) if LAB else np.fromfile("target_set.data", np.uint8)
val_set = np.fromfile("Shape_val_set.data", np.float32) if shape else np.fromfile("val_set.data", np.float32)
val_target_set = np.fromfile("LAB_val_target_set.data", np.uint8) if LAB else np.fromfile("val_target_set.data", np.uint8)
test_set = np.fromfile("Shape_test_set.data", np.float32) if shape else np.fromfile("test_set.data", np.float32)
test_target_set = np.fromfile("LAB_test_target_set.data", np.uint8) if LAB else np.fromfile("test_target_set.data", np.uint8)

master = open("master.txt")
num_train = int(master.readline())
num_val = int(master.readline())
num_test = int(master.readline())

if K.image_data_format() == 'channels_first':
    train_set = train_set.reshape(num_train, 1, img_rows, img_cols)
    target_set = target_set.reshape(num_train, 3, img_rows, img_cols)
    val_set = val_set.reshape(num_val, 1, img_rows, img_cols)
    val_target_set = val_target_set.reshape(num_val, 3, img_rows, img_cols)
    test_set = test_set.reshape(num_test, 1, img_rows, img_cols)
    test_target_set = test_target_set.reshape(num_test, 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_set = train_set.reshape(num_train, img_rows, img_cols, 1)
    target_set = target_set.reshape(num_train, img_rows, img_cols, 3)
    val_set = val_set.reshape(num_val, img_rows, img_cols, 1)
    val_target_set = val_target_set.reshape(num_val, img_rows, img_cols, 3)
    test_set = test_set.reshape(num_test, img_rows, img_cols, 1)
    test_target_set = test_target_set.reshape(num_test, img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 1)


target_set = target_set/255
test_target_set = test_target_set/255

# evaluate loaded model on test data
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy', MSEMetric, MSEVar, PSNRMetric, PSNRVar, SSIMMetric, SSIMVar])

cv2.namedWindow('Disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Disp', 900,300)

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
rect = dlib.rectangle(0, 0, 128, 128)


num = 0
run = True

while run:
    print("Get pic num:")
    pic = int(input())
    col = np.squeeze(model.predict(np.expand_dims(test_set[pic], axis=0)), axis=0)
    if(LAB):
        col = np.floor(col*255).astype(np.uint8)
        col = cv2.cvtColor(col, cv2.COLOR_Lab2BGR)/255
    inp = cv2.cvtColor(test_set[pic], cv2.COLOR_GRAY2BGR)
    if(LAB):
        truth = np.floor(test_target_set[pic]*255).astype(np.uint8)
        truth = cv2.cvtColor(truth, cv2.COLOR_Lab2BGR)/255
    else:
        truth = test_target_set[pic]

    shape_true = predictor(cv2.cvtColor((truth*255).astype(np.uint8), cv2.COLOR_BGR2GRAY), rect)
    shape_true = shape_to_np(shape_true)
    shape_pred = predictor(cv2.cvtColor((col*255).astype(np.uint8), cv2.COLOR_BGR2GRAY), rect)
    shape_pred = shape_to_np(shape_pred)
    diff = np.zeros((128,128,3))
    t_overlay = np.zeros((128, 128, 3))
    c_overlay = np.zeros((128, 128, 3))
    for i in range(68):
        cv2.circle(c_overlay, (shape_pred[i,0], shape_pred[i,1]), 1, (204,0,204), -1)
        cv2.circle(diff, (shape_pred[i,0], shape_pred[i,1]), 1, (204,0,204), -1)
        cv2.circle(t_overlay, (shape_true[i,0], shape_true[i,1]), 1, (0,204,204), -1)
        cv2.circle(diff, (shape_true[i,0], shape_true[i,1]), 1, (0,204,204), -1)
        dist = math.sqrt((shape_true[i,0]-shape_pred[i,0])*(shape_true[i,0]-shape_pred[i,0]) + (shape_true[i,1]-shape_pred[i,1])*(shape_true[i,1]-shape_pred[i,1]))
        color = dist/5 if dist <= 5 else 1.0
        cv2.line(diff, (shape_pred[i,0], shape_pred[i,1]), (shape_true[i,0], shape_true[i,1]), (0,255*(1-color),255*color), 1)

    truth = cv2.addWeighted(t_overlay/255, 0.3, truth, 1.0, 0)
    col = cv2.addWeighted(c_overlay/255, 0.3, col, 1.0, 0)
    cv2.imshow("Disp", np.hstack((inp, col, truth, diff)))
    if cv2.waitKey(0) == ord("c"):
        print("cont")
        continue

    if cv2.waitKey(0) == ord("s"):
        print("save")
        cv2.imwrite("Manual\\Auto" + str(num) + ".png", np.hstack((inp*255, col*255, truth*255, diff)))
        num += 1
        continue

    if cv2.waitKey(0) == ord("q"):
        print("break")
        run = False
        continue