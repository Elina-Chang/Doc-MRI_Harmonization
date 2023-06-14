import os

from Tools.RootPath2CaseList import RootPath2CaseList
from Tools.Nii2Npy import Nii2Npy
from Tools.normalization1.new_OtsuSegment import new_OtsuSegment


def get_file_list(root_path, file_name):
    case_list = RootPath2CaseList(root_path)
    file_list = [os.path.join(case_name, file_name) for case_name in case_list]
    return file_list


def get_data3d_cube(file_list):
    for path in file_list:
        img, data = Nii2Npy(path)
        data = crop_center3D(data, 280, 280)
        data3d_cube.extend()
    data3d_cube = [Nii2Npy(path) for path in file_list]
    data2d_cube = [[data3d[i] for i in range(data3d.shape[0])] for data3d in data3d_cube]
    return data2d_cube


def get_pixel_list(root_path, file_name):
    file_list = get_file_list(root_path, file_name)
    data2d_cube = get_data2d_cube(file_list)
