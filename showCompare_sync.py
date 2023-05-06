import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import open3d as o3d

with open('D:\git\kitti-velo2cam\calib.txt','r') as f:
    calib = f.readlines()

with open('D:\data\sync_list.txt','r') as f:
    sync_list = f.readlines()

img_org = 'D:\\data\\fusion_data\\images\\'
pcd_org = 'D:\\data\\fusion_data\\pcds\\'
interp_org = 'D:\data\subdata\\'

def velo2cam(source_img_path, sync_img_path, pcd_path):
    infer = np.array([float(x) for x in calib[1].strip('\n').split(' ')[1:]]).reshape(3,3)
    Tr_velo_to_cam = np.array([float(x) for x in calib[3].strip('\n').split(' ')[1:]]).reshape(3,4)
    # read raw data from binary
    points=open_points(pcd_path)

    velo = np.insert(points,3,1,axis=1).T
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    cam = infer.dot(Tr_velo_to_cam.dot(velo))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    plt.figure(figsize=(13,8),dpi=150,tight_layout=True)
    source_img = mpimg.imread(source_img_path)
    source_img = (source_img*255).astype(np.uint8)
    sync_img = mpimg.imread(sync_img_path)
    IMG_H,IMG_W,_ = source_img.shape
    compare_img = np.zeros((IMG_H, IMG_W*2, 3))
    compare_img[:, :IMG_W, :] = source_img.copy()
    compare_img[:, IMG_W:, :] = sync_img.copy()
    compare_img = np.array(compare_img, dtype=np.uint8)
    # restrict canvas in range
    plt.axis([0, 2*IMG_W, IMG_H, 0])
    plt.imshow(compare_img)
    # filter point out of canvas
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    # generate color map from depth
    u,v,z = cam
    plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
    plt.scatter([u + IMG_W],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
    plt.show()

def get_timestamp(pcd_list, index):
    return float(pcd_list[index].split('_')[3][0:-4])

def open_points(pcdata_name):
    point_cloud=o3d.io.read_point_cloud(pcdata_name)
    points = np.asarray(point_cloud.points) 
    # filting unconcerned points
    points_filted = []
    for point in points:
        if point[0] > 0 and point[0] < 60 and point[1] > -8 and point[1] < 12:
            points_filted.append(point)
    points = np.asarray(points_filted)
    return points

if __name__ == "__main__":
    img_list = os.listdir(img_org)
    pcd_list = os.listdir(pcd_org)
    for index in range(len(sync_list)):
        img = img_org + img_list[index]
        pcd = pcd_org + pcd_list[index]
        sync = interp_org + sync_list[index].replace("\n", "")
        velo2cam(img, sync, pcd)
            