import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import open3d as o3d

# sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
# name = '%06d'%sn # 6 digit zeropadding
img = 'D:\data\\test01.jpg'
binary = 'D:\\data\\fusion_data\\pcds\\rslidar_points_128_1667373209.450156927.pcd'
with open('D:\git\Sync_pcd_data\calib.txt','r') as f:
    calib = f.readlines()

infer = np.array([float(x) for x in calib[1].strip('\n').split(' ')[1:]]).reshape(3,3)
Tr_velo_to_cam = np.array([float(x) for x in calib[3].strip('\n').split(' ')[1:]]).reshape(3,4)
# read raw data from binary
point_cloud=o3d.io.read_point_cloud(binary)
points = np.asarray(point_cloud.points)
velo = np.insert(points,3,1,axis=1).T
velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
print(velo.shape)
cam = infer.dot(Tr_velo_to_cam.dot(velo))
print(cam.shape)
cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
# get u,v,z
u1, v1, z1 = cam
cam[:2] /= cam[2,:]
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
# plt.imshow(png)
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)
# filter point out of canvas
u,v,z = cam
print(z.shape)
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)
# generate color map from depth
u,v,z = cam
plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
plt.savefig("fig2.jpg")
plt.show()
