import pyrealsense2 as rs
import cv2
import numpy as np
import dlib
import keras
from keras.models import model_from_json

LAB = True


def colorize(detector, dimage, darray):
    rects = detector(dimage)
    if len(rects) != 0:
        rect = rects[0]
        small_depth = darray[rect.top():rect.bottom(), rect.left():rect.right()]
        try:
            small_depth = cv2.resize(small_depth, (128,128))
            cv2.threshold(small_depth, 1000, cv2.THRESH_BINARY, cv2.THRESH_TOZERO_INV, small_depth)
            small_depth = small_depth - np.average(small_depth)
            small_depth = small_depth.astype(np.float32)
            cv2.normalize(small_depth, small_depth, 0, 1, cv2.NORM_MINMAX)
            inferred = np.squeeze(model.predict(np.expand_dims(np.expand_dims(small_depth, axis=0), axis=3)), axis=0)
            if LAB:
                inferred = np.floor(inferred * 255).astype(np.uint8)
                inferred = cv2.cvtColor(inferred, cv2.COLOR_LAB2BGR)/255
            inferred = cv2.resize(inferred, (rect.width()-1, rect.height()-1))
            cv2.imshow("small", inferred)
            dimage[rect.top():rect.bottom(), rect.left():rect.right()] = inferred*255

        except cv2.error as e:
            print("No face found!")

cv2.namedWindow("Disp", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Disp", 1280, 480)

cv2.namedWindow("small", cv2.WINDOW_NORMAL)
cv2.resizeWindow("small", 256, 256)

cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480);
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe = rs.pipeline()
pipe.start(cfg)

spat_fil = rs.spatial_filter()
spat_fil.set_option(rs.option.filter_magnitude, 2.0)
spat_fil.set_option(rs.option.filter_smooth_alpha, 0.5)
spat_fil.set_option(rs.option.filter_smooth_delta, 20.0)

temp_fil = rs.temporal_filter()
temp_fil.set_option(rs.option.filter_smooth_alpha, 0.4)
temp_fil.set_option(rs.option.filter_smooth_delta, 20.0)

colorizer = rs.colorizer()

detector = dlib.fhog_object_detector("depth_detector.svm")

json_file = open('large_model.json', 'r')
loaded_json = json_file.read()
json_file.close()
model = model_from_json(loaded_json)
model.load_weights("large_model.h5")

run = True

while run:
    frames = pipe.wait_for_frames()
    depth = frames.get_depth_frame()
    depth = spat_fil.process(depth)
    ##depth = temp_fil.process(depth)
    depth_data = depth.as_frame().get_data()
    darray = np.asanyarray(depth_data)
    depth_col = colorizer.process(depth)
    depth_col_data = depth_col.as_frame().get_data()
    dimage = np.asanyarray(depth_col_data)
    color = frames.get_color_frame()
    color_data = color.as_frame().get_data()
    col = np.asanyarray(color_data)
    col = cv2.cvtColor(col, cv2.COLOR_RGB2BGR)
    colorize(detector, dimage, darray)

    cv2.imshow("Disp", np.hstack((col, dimage)))
    if cv2.waitKey(1) == ord("q"):
        run = False