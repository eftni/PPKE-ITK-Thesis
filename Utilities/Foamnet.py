from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import cv2
import dlib


##tf.enable_eager_execution()

img_rows, img_cols = 128, 128

demo = False
demo_name = "Dropout"
LAB = True
shape = True
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


json_file = open('large_model.json', 'r')
loaded_json = json_file.read()
json_file.close()
model = model_from_json(loaded_json)
model.load_weights("large_model.h5")

foam_set = np.fromfile("foam_set.data", np.float32)

master = open("master.txt")
num_train = int(master.readline())
num_val = int(master.readline())
num_test = int(master.readline())

print(num_train, num_val, num_test)

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
    foam_set = foam_set.reshape(10, 1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)
    foam_set = foam_set.reshape(10, img_rows, img_cols, 1)

# evaluate loaded model on test data
##model.compile()


cv2.namedWindow('Disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Disp', 900,300)
cv2.namedWindow('Colorized', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Colorized', 1280,480)

ROIs = open("C:\\Users\\Niko\\Desktop\\Dataset\\Data_saver\\ROIs\\foam_ROIs.txt")
rects = [line.rstrip('\n') for line in ROIs]

for i in range(0, 10):
    print(str(i))
    col = np.squeeze(model.predict(np.expand_dims(foam_set[i], axis=0)), axis=0)
    if(LAB):
        col = np.floor(col*255).astype(np.uint8)
        col = cv2.cvtColor(col, cv2.COLOR_Lab2BGR)/255
    input = cv2.cvtColor(foam_set[i], cv2.COLOR_GRAY2BGR)
    cv2.imshow("Disp", np.hstack((input, col)))

    params = [int(param) for param in rects[i].split(' ')]
    truth = cv2.imread("C:\\Users\\Niko\\Desktop\\Dataset\\Data_saver\\color\\foam_col" + str(i) + ".png")
    base = cv2.imread("C:\\Users\\Niko\\Desktop\\Dataset\\Data_saver\\depth_color\\foam_col_depth" + str(i) + ".png")
    col = cv2.resize(col, (params[2], params[3]))
    base[params[1]:params[1]+params[3], params[0]:params[0]+params[2],:] = col*255
    cv2.imshow("Colorized", np.hstack((truth, base)))
    cv2.imwrite("C:\\Users\\Niko\\Desktop\\foamlord.png", np.hstack((truth, base)))
    if autoplay:
        cv2.waitKey(50)
    else:
        cv2.waitKey(0)