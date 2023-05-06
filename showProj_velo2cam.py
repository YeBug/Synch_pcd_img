import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import open3d as o3d

with open('D:\git\kitti-velo2cam\calib.txt','r') as f:
    calib = f.readlines()

img_org = 'D:\\data\\fusion_data\\images\\'
pcd_org = 'D:\\data\\fusion_data\\pcds\\'

def velo2cam(img_path, pcd_path):
    infer = np.array([float(x) for x in calib[1].strip('\n').split(' ')[1:]]).reshape(3,3)
    print(infer)
    Tr_velo_to_cam = np.array([float(x) for x in calib[3].strip('\n').split(' ')[1:]]).reshape(3,4)
    print(Tr_velo_to_cam)
    # read raw data from binary
    points=open_points(pcd_path)

    velo = np.insert(points,3,1,axis=1).T
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    cam = infer.dot(Tr_velo_to_cam.dot(velo))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    plt.figure(figsize=(12,7),dpi=120,tight_layout=True)
    png = mpimg.imread(img_path)
    IMG_H,IMG_W,_ = png.shape
    # restrict canvas in range
    plt.axis([0,IMG_W,IMG_H,0])
    plt.imshow(png)
    # filter point out of canvas
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    # generate color map from depth
    u,v,z = cam
    plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
    plt.show()

def get_timestamp(pcd_list, index):
    return float(pcd_list[index].split('_')[3][0:-4])

def open_points(pcdata_name):
    point_cloud=o3d.io.read_point_cloud(pcdata_name)
    points = np.asarray(point_cloud.points) 
    points_filted = []
    for point in points:
        if point[0] > 0 and point[0] < 60 and point[1] > -8 and point[1] < 12:
            points_filted.append(point)
    points = np.asarray(points_filted)
    return points

if __name__ == "__main__":
    img_list = os.listdir(img_org)
    pcd_list = os.listdir(pcd_org)
    index = 0
    pre_lack = 10000
    for img_path in img_list:
        if img_path[6] != '0':
            break
        img_time = float(img_path.split('_')[1][0:-4])
        while index < len(pcd_list):
            pcd_time = get_timestamp(pcd_list, index)
            next_index = index + 1
            if next_index == len(pcd_list):
                next_index = 0
            cur_lack = abs(img_time-pcd_time)
            next_lack = abs(img_time-get_timestamp(pcd_list, next_index))
            if next_lack < cur_lack:
                index = next_index
            else:
                break
        print("{} ==> {}".format(img_path, pcd_list[index]))
        img = img_org + img_path
        pcd = pcd_org + pcd_list[index]
        velo2cam(img, pcd)
            