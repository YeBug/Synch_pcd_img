# Sync_pcd_data

## Description
### calib.txt
    calib matrix for camera
### colorbin.py
    filtes pcd points and pcd segmentation labels out of image range
    projects pcd points on 2D images
    counts error points on road segmentation area
### imgseg.py
    using cityscapes image segmentation model in transformers lib
    segments interpolation images
### showComapre_sync.py
    projects pcd points on origin images and interpolation images sync for pcd data
    compare those projection results
### showProj_velo2cam.py
    util script for exploring image data with projected pcd points
### synchronize_pcdimg.py
    pipeline for synchronizing image data and pcd data
    1. get pcd data list
    2. for each pcd data, N interpolation images are created, N = 10 defaulted
    3. segment interpolation images and count error points with projected pcd data
    4. select image with minimum error number as sync result
