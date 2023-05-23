import os
import sys
import cv2
from tqdm import tqdm
from colorbin import proj2img, get_proj_pcd
from imgseg import model_init, img_seg

model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
# seg_img_path = "D:\data\subdata\\"
# pcd_path = "D:\\data\\fusion_data\\pcds\\"
# seg_label_path = "D:\git\SqueezeSegV3\sample_output\sequences\\00\predictions\\"
vfi_img_path = "D:\\data\\testdata\subdata\\"
pcd_path = "D:\\data\\testdata\\pcds\\"
seg_label_path = "D:\\data\\testdata\\predictions\\"
vfi_seg_path =  "D:\\data\\testdata\\color_seg\\"
calib_path = 'calib.txt'


if __name__ == "__main__":
    img_list = os.listdir(vfi_img_path)
    img_list.sort()
    pcd_list = os.listdir(pcd_path)
    interp_list = os.listdir(vfi_img_path)
    interp_index = 0
    # cache list save interp_seg data before pcd_time
    cache_list = [[None, None] for i in range(9)]
    for index in range(len(pcd_list)):
        pcd_data_path = pcd_path+pcd_list[index]
        label_data_path = seg_label_path+pcd_list[index].replace(".pcd", ".label")
        cam, label = get_proj_pcd(calib_path, label_data_path, pcd_data_path)

        feature_extractor, model = model_init(model_name)
        max_rate = 0

        # trace cache list, time before pcd time
        cache_index = 0
        for i in range(9):
            if not cache_list[i][0]:
                break 
            cache_path = cache_list[i][0]
            cache_seg = cache_list[i][1]
            acc_rate = proj2img(cam, label, cache_seg)
            if max_rate < acc_rate:
                best_path = cache_path
                max_rate = acc_rate

        # trace interp img, time after pcd time
        for i in range(9):
            if interp_index >= len(interp_list):
                break
            interp_path = vfi_img_path+"\\"+interp_list[interp_index]
            color_seg = img_seg(feature_extractor, model, interp_path)
            cache_list[i][0] = interp_list[interp_index]
            cache_list[i][1] = color_seg
            acc_rate = proj2img(cam, label, color_seg)
            if max_rate < acc_rate:
                best_path = interp_list[interp_index]
                max_rate = acc_rate
            interp_index += 1
        print(best_path)
            
        