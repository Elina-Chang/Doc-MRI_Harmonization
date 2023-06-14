import SimpleITK as sitk
import numpy as np
import random
import os
import glob

from branches.check_img_roi.Visualization import Imshow3DArray


# 写一个获取所有img/roi绝对路径的函数
def get_all_ABSpaths(root_path, file_name):
    cases_ABSpaths = glob.glob(root_path + "\*")
    files_ABSpaths = []
    for case_ABSpath in cases_ABSpaths:
        file_ABSpath = os.path.join(case_ABSpath, file_name)
        files_ABSpaths.append(file_ABSpath)

    return files_ABSpaths


# 定义一个用来display重采样之后的img和roi的函数
def display_img_roi(root_path, img_file_name, roi_file_name):
    # # 生成一个[,]内的随机数
    # rand = random.randint(0, 65)
    # print(rand)
    all_imgs_ABSpaths = get_all_ABSpaths(root_path=root_path, file_name=img_file_name)  # [rand:]
    all_rois_ABSpaths = get_all_ABSpaths(root_path=root_path, file_name=roi_file_name)  # [rand:]
    for img_ABSpath, roi_ABSpath in zip(all_imgs_ABSpaths, all_rois_ABSpaths):
        print(img_ABSpath.split(sep="\\")[-2])
        img_data = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(img_ABSpath)), [1, 2, 0])
        roi_data = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(roi_ABSpath)), [1, 2, 0])
        # img_data = np.flipud(img_data)
        # roi_data = np.flipud(roi_data)
        # roi_data = np.fliplr(roi_data)
        print(img_data.min(), img_data.max(), img_data.dtype)
        print(roi_data.min(), roi_data.max(), roi_data.dtype)
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        roi_data = (roi_data - roi_data.min()) / (roi_data.max() - roi_data.min())
        Imshow3DArray(data=img_data, roi=roi_data)


def main():
    root_path = r"E:\PhD\Data_renji\GE_3T_resampled"
    img_file_name = r"t2sag_resampled.nii"
    roi_file_name = r"womb_roi_resampled.nii"
    display_img_roi(root_path, img_file_name, roi_file_name)


if __name__ == "__main__":
    main()
