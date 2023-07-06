import numpy as np

img = 'D:\data\\test01.jpg'
with open('D:\git\Sync_pcd_data\calib.txt','r') as f:
    calib = f.readlines()

infer = np.array([float(x) for x in calib[1].strip('\n').split(' ')[1:]]).reshape(3,3)
Tr_velo_to_cam = np.array([float(x) for x in calib[3].strip('\n').split(' ')[1:]]).reshape(3,4)
Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))

x, y, d = [], [], []
u = (x - infer[0][2]) / infer[0][0]
v = (y - infer[1][2]) / infer[1][1]

coords = np.array([u, v, 1])
cam = np.linalg.inv(infer) @ coords * d
world = Tr_velo_to_cam @ np.hstack((cam, 1))