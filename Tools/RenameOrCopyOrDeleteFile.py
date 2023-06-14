import numpy as np
import os
import shutil

from Tools.RootPath2CaseList import RootPath2CaseList


# 创建一个重命名函数: os.rename(src，dst)
# 传入每一个file的名字，然后重命名为"xxx.nii"file
# def rename_file(root_path, dst_filename):
#     cases_ABSpaths = glob.glob(root_path + "\*")
#     for case_ABSpath in cases_ABSpaths:
#         src = glob.glob(case_ABSpath + "\*t2sag_ori.nii.gz")[0]
#         dst = os.path.join(case_ABSpath, dst_filename)
#         os.rename(src, dst)


def rename_file(root_path="", src_filename="", dst_filename=""):
    case_list = RootPath2CaseList(root_path)
    for case in case_list:
        src = case + "\\" + src_filename
        dst = case + "\\" + dst_filename
        os.rename(src, dst)


def copy_file(src_root_path="", src_filename="", dst_root_path="", dst_filename=""):
    case_list = RootPath2CaseList(src_root_path)
    for case in case_list:
        src = case + "\\" + src_filename
        dst = dst_root_path + "\\" + case.split("\\")[-1] + "\\" + dst_filename
        if not os.path.exists(dst_root_path + "\\" + case.split("\\")[-1]):
            os.makedirs(dst_root_path + "\\" + case.split("\\")[-1])
        shutil.copy2(src, dst)


def delete_file(root_path="", file_name=""):
    case_list = RootPath2CaseList(root_path)
    for case in case_list:
        data_file = case + "\\" + file_name
        # 如果类型是文件则进行删除
        if os.path.isfile(data_file):
            os.remove(data_file)
        else:
            print(f'Error: {data_file} not a valid filename')


# 1. 传入根目录，和目标名称
def rename_main():
    root_path = r"G:\PhD\Data_renji\Data_forFeatureExtractor\GE_3T"
    rename_file(root_path, src_filename="t2sag_styled.nii", dst_filename=r"t2sag_dualGAN.nii")


def copy_main():
    copy_file(src_root_path=r"G:\PhD\Data_renji\Data_forFeatureExtractor_\39SIEMENS",
              src_filename="cervix_roi.nii.gz",
              dst_root_path=r"G:\PhD\Data_renji\Data_forFeatureExtractor\39SIEMENS",
              dst_filename=r"cervix_roi.nii.gz")


def delete_main():
    delete_file(root_path=r"G:\PhD\Data_renji\Data_forFeatureExtractor\GE_3T", file_name="t2sag_matched.nii")


if __name__ == "__main__":
    copy_main()
