import glob
import os
import SimpleITK as sitk


# 写一个获取所有img/roi绝对路径的函数
def get_all_ABSpaths(root_path, file_name):
    cases_ABSpaths = glob.glob(root_path + "\*")
    files_ABSpaths = []
    for case_ABSpath in cases_ABSpaths:
        file_ABSpath = os.path.join(case_ABSpath, file_name)
        files_ABSpaths.append(file_ABSpath)

    return files_ABSpaths


# 统计每个case里面的image和roi的分辨率信息
# 写一个统计分辨率的函数
def resolution_statistic(images_paths):
    resolution_list = []
    for image_path in images_paths:
        image = sitk.ReadImage(image_path)
        resolution = image.GetSpacing()
        resolution_list.append(resolution)

    return resolution_list  # 用于存放所有的分辨率


# 写一个取出所有X轴方向分辨率的函数
def get_Xresolution(resolution_list):
    Xresolution_list = [resolution_list[i][0] for i in range(len(resolution_list))]

    return Xresolution_list


# 写一个统计所有image X轴分辨率信息的函数
def img_Xresolution(imgs_paths_list):
    img_resolution_list = resolution_statistic(imgs_paths_list)
    img_Xresolution_list = get_Xresolution(img_resolution_list)

    return img_Xresolution_list


# 写一个统计所有roi X轴分辨率信息的函数
def roi_Xresolution(rois_paths_list):
    roi_resolution_list = resolution_statistic(rois_paths_list)
    roi_Xresolution_list = get_Xresolution(roi_resolution_list)

    return roi_Xresolution_list


# 先来测试一下能不能正常获取X轴的分辨率信息
def main():
    root_path = r"E:\PhD\Data_renji\GE_3T_resmapled_renamed_test"
    file_name = r"t2sag_resampled.nii"
    all_ABSpaths = get_all_ABSpaths(root_path, file_name)
    img_Xresolution_list = img_Xresolution(all_ABSpaths)

    return img_Xresolution_list


if __name__ == "__main__":
    img_Xresolution_list = main()
    print(len(img_Xresolution_list))
    import matplotlib.pyplot as plt
    plt.hist(img_Xresolution_list)
    plt.show()
