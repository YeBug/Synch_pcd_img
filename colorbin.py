import numpy as np
import open3d as o3d

def get_calib(calib_name):
    with open(calib_name,'r') as f:
        calib = f.readlines()
    infer = np.array([float(x) for x in calib[1].strip('\n').split(' ')[1:]]).reshape(3,3)
    Tr_velo_to_cam = np.array([float(x) for x in calib[3].strip('\n').split(' ')[1:]]).reshape(3,4)
    return infer, Tr_velo_to_cam

def open_label(label_name):
    label = np.fromfile(label_name, dtype=np.int32)
    return label

def open_points(pcdata_name):
    point_cloud=o3d.io.read_point_cloud(pcdata_name)
    points = np.asarray(point_cloud.points) 
    points_filted = []
    for point in points:
        if point[0] > 0 and point[0] < 60 and point[1] > -8 and point[1] < 12:
            points_filted.append(point)
    points = np.asarray(points_filted)
    return points

def get_proj_pcd(calib_name, label_name, pcdata_name):
    infer, Tr_mat = get_calib(calib_name)
    label = open_label(label_name)
    points = open_points(pcdata_name)
    velo = np.insert(points,3,1,axis=1).T
    label = label.T
    label = np.delete(label,np.where(velo[0,:]<0))
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    cam = infer.dot(Tr_mat.dot(velo))
    label = np.delete(label,np.where(cam[2,:]<0))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
    cam[:2] /= cam[2,:]
    # do projection staff
    IMG_H,IMG_W, = 1080, 1920
    # restrict canvas in range

    # filter point out of canvas
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    label = np.delete(label,np.where(outlier))
    return cam, label

def proj2img(cam, label, color_seg):
    IMG_H, IMG_W, = 1080, 1920
    # generate color map from depth
    u,v,z = cam
    count_error = 0
    target_points = 0
    for i in range(len(label)):
        if label[i] == 40 or label[i] ==44  or label[i] == 49 or label[i] == 48:
            label[i] = 40 
        else:
            label[i] = 12
            target_points += 1
            if list(color_seg[int(v[i]), int(u[i])]) == [128, 64, 128]:
                count_error += 1
    acc_rate = (target_points - count_error) / target_points
    return acc_rate

if __name__ == "__main__":
    img_path = 'D:\data\\test0.jpg'
    calib_path = 'D:\git\kitti-velo2cam\calib.txt'
    label_path = "D:\git\SqueezeSegV3\sample_output\sequences\\00\predictions\\rslidar_points_128_1667373209.350169897.label"
    pcdata_path = 'D:\\data\\fusion_data\\pcds\\rslidar_points_128_1667373209.350169897.pcd'
    proj2img(calib_path, label_path, pcdata_path, ["test0.jpg"], "D:\data\\")