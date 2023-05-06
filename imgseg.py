from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import os
import cv2
import numpy as np

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]

model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
img_org = 'D:\\data\\fusion_data\\images\\'
seg_save = 'D:\LocalCode\pcd\segimgs\\'
device = "cuda:0"

def model_init(model_name):
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    return feature_extractor, model

def img_seg(feature_extractor, model, img_path):
    image = Image.open(img_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    seg_map = SegformerFeatureExtractor.post_process_semantic_segmentation(self=None, outputs=outputs, target_sizes=[[1080, 1920]])
    seg_map = seg_map[0].numpy()
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(PALETTE):
        color_seg[seg_map == label, :] = color
    color_seg = color_seg[..., ::-1]
    return color_seg


if __name__ == "__main__":
    img_list = os.listdir(img_org)
    feature_extractor, model = model_init(model_name)
    for img_path in img_list:
        if img_path.startswith("camera0"):
            print(img_org+img_path)
            color_seg = img_seg(feature_extractor, model, img_org+img_path)
            cv2.imwrite(seg_save+img_path, color_seg)