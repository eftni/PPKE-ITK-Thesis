import csv
import matplotlib.pyplot as plt
import numpy as np
import math

def load():
    pose_truth = {'x': [], 'y': [], 'z': []}
    pose_pred = {'x': [], 'y': [], 'z': []}
    landmarks_truth = {'x': [], 'y': []}
    landmarks_pred = {'x': [], 'y': []}
    truth = open("Truth.csv")
    csv_reader = csv.DictReader(truth, delimiter=',')
    rownum = 0
    for row in csv_reader:
        pose_truth['x'].append(float(row[' pose_Tx']))
        pose_truth['y'].append(float(row[' pose_Ty']))
        pose_truth['z'].append(float(row[' pose_Tz']))
        tmpx = []
        tmpy = []
        for i in range(0, 68):
            tmpx.append(float(row[' x_' + str(i)]))
            tmpy.append(float(row[' x_' + str(i)]))
        landmarks_truth['x'].append(tmpx)
        landmarks_truth['y'].append(tmpy)
        rownum = rownum + 1
    pred = open("Test.csv")
    csv_reader = csv.DictReader(pred, delimiter=',')
    rownum = -1
    for row in csv_reader:
        pose_pred['x'].append(float(row[' pose_Tx']))
        pose_pred['y'].append(float(row[' pose_Ty']))
        pose_pred['z'].append(float(row[' pose_Tz']))
        tmpx = []
        tmpy = []
        for i in range(0, 68):
            tmpx.append(float(row[' x_' + str(i)]))
            tmpy.append(float(row[' x_' + str(i)]))
        landmarks_pred['x'].append(tmpx)
        landmarks_pred['y'].append(tmpy)
        rownum = rownum + 1
    return pose_truth, pose_pred, landmarks_truth, landmarks_pred


def ew_diff_calc(pose_truth, pose_pred, landmarks_truth, landmarks_pred):
    pose_truth['x'] = pose_truth['x'][0:442]
    pose_truth['y'] = pose_truth['y'][0:442]
    pose_truth['z'] = pose_truth['z'][0:442]
    pose_diff = np.sqrt(np.square(np.subtract(pose_truth['x'],pose_pred['x'])) + np.square(np.subtract(pose_truth['y'],pose_pred['y'])) + np.square(np.subtract(pose_truth['z'],pose_pred['z'])))
    landmark_diff = []
    for i in range(0, 441):
        #landmarks_truth[i]['x'] = landmarks_truth[i]['x'][]
        landmark_diff.append(np.sqrt(np.square(np.subtract(landmarks_truth['x'][i],landmarks_pred['x'][i])) + np.square(np.subtract(landmarks_truth['y'][i],landmarks_pred['y'][i]))))
    return pose_diff, landmark_diff


pose_truth, pose_pred, landmarks_truth, landmarks_pred = load()
pose_diff, landmark_diff = ew_diff_calc(pose_truth, pose_pred, landmarks_truth, landmarks_pred)
fig = plt.figure()
plt.plot(pose_diff)
print(np.average(pose_diff))
avg = [np.average(pose_diff)] * 442
plt.plot(avg)
plt.show()

fig = plt.figure()
for i in range(68):
    plt.plot(landmark_diff[:,i])
plt.plot(avg)
plt.show()