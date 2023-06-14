import matplotlib.pyplot as plt
import SimpleITK as sitk
import skimage.transform as transform
import numpy as np
import os
import glob
import pandas as pd


def find_modal(cases_path):
    """

    :param cases_path: a list contains the absolute paths of all cases in a certain DIR
    :return: a list contains needed certain modals
    """
    modals_path = []
    for case_path in cases_path:
        modal_path = glob.glob(case_path + r"/*t2sag.nii")
        modals_path.extend(modal_path)
    return sorted(modals_path)


def get_abnormal_cases(file_path):
    """
    # Acquire the cases' names of the abnormal data
    :param path: .excel file path for storing the abnormal info
    :return: (set)形式: {}
    """
    csv = pd.read_excel(file_path, index_col=None)
    GE_abnormal_cases = set(csv.iloc[:, 0].dropna())
    Philips_abnormal_cases = set(csv.iloc[:, 1].dropna())
    return GE_abnormal_cases, Philips_abnormal_cases


def get_smallFOV_cases(file_path):
    df = pd.read_excel(file_path, index_col=None)
    cases_list = df["Case"].values.tolist()
    return cases_list


def get_selected_path(root_dir=r"G:\PhD\Data_renji\Data_3T_Norm1", dir1="GE_3T_Norm1", dir2="Philips_Norm1",
                      only_3T=False, Gfile_path="G:\PhD\Data_renji\GE_data_info.csv",
                      elinimate_smallFOV=True, Pfile_path=r"G:\PhD\Data_renji\Philips_data_info.xls",
                      elinimate_abnormal=False, abnormal_cases_csv=r"G:\PhD\Data_renji\Abnormal_case.xls"):
    """
    # Return selected image paths
    :param root_dir:
    :param dir1:
    :param dir2:
    :param GE_csv_path:
    :return:
    """
    # 得到所有的dir1里面的cases和dir2里面的cases
    cases_dir1 = os.listdir(os.path.join(root_dir, dir1))
    # print(len(cases_dir1), cases_dir1)
    cases_dir2 = os.listdir(os.path.join(root_dir, dir2))
    # print(len(cases_dir2), cases_dir2)

    # 读取包含GE数据扫描信息的.csv文件，并存储1.5T的cases
    GE_data_info = pd.read_csv(Gfile_path, index_col=False)
    cases_list = []  # 用于存放1.5T的cases
    for row_id in range(GE_data_info.shape[0]):
        row = list(GE_data_info.iloc[row_id, :])
        # 筛选cases
        if "b'1.5T" in row[1]:
            cases_list.append(row[0])
    # print(len(cases_list), cases_list)

    # 如有必要，剔除1.5T数据
    if only_3T:
        cases_dir1 = list(set(cases_dir1).difference(set(cases_list)))  # 在cases_dir1中而不在cases_list中，即剔除1.5T数据
    # print(len(cases_dir1), cases_dir1)

    # 如有必要，剔除异常cases
    if elinimate_abnormal:
        # 获取异常cases
        GE_abnormal_cases, Philips_abnormal_cases = get_abnormal_cases(file_path=abnormal_cases_csv)  # 请注意，这里是以集合形式存储
        # print(len(GE_abnormal_cases))
        # 剔除异常cases
        cases_dir1 = list(set(cases_dir1).difference(GE_abnormal_cases))
        cases_dir2 = list(set(cases_dir2).difference((Philips_abnormal_cases)))
    # print(len(cases_dir1), len(cases_dir2))

    # 如有必要，剔除smallFOV cases
    if elinimate_smallFOV:
        # 获取smallFOV cases
        Philips_smallFOV_cases = get_smallFOV_cases(file_path=Pfile_path)  # 请注意，这里是以集合形式存储
        # print(len(Philips_smallFOV_cases))
        # 剔除smallFOV cases
        cases_dir2 = list(set(cases_dir2).difference((Philips_smallFOV_cases)))
    # print(len(cases_dir1), len(cases_dir2))

    # 绝对路径
    cases_dir1_path = [os.path.join(root_dir, dir1, case_path) for case_path in cases_dir1]
    # print(len(cases_dir1_path),cases_dir1_path)
    cases_dir2_path = [os.path.join(root_dir, dir2, case_path) for case_path in cases_dir2]
    # print(len(cases_dir2_path),cases_dir2_path)

    modals_dir1 = np.squeeze(find_modal(cases_dir1_path))
    modals_dir2 = np.squeeze(find_modal(cases_dir2_path))
    # print(len(modals_dir1), len(modals_dir2))
    # print(modals_dir1)
    # print(modals_dir2)

    return modals_dir1, modals_dir2


def store3dData2list():
    GE_data_path, Philips_data_path = get_selected_path()
    GE_3dData_list = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in GE_data_path]
    Philips_3dData_list = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in Philips_data_path]

    return GE_3dData_list, Philips_3dData_list


def store2dData2list():
    GE_3dData_list, Philips_3dData_list = store3dData2list()
    GE_2dDataTempList = [[x3d[i] for i in range(x3d.shape[0])] for x3d in GE_3dData_list]
    Philips_2dDataTempList = [[x3d[i] for i in range(x3d.shape[0])] for x3d in Philips_3dData_list]
    from functools import reduce
    GE_2dData_list = reduce(lambda x, y: x + y, GE_2dDataTempList)
    Philips_2dData_list = reduce(lambda x, y: x + y, Philips_2dDataTempList)

    return GE_2dData_list, Philips_2dData_list


def zeropadding2D(img_data, target_size):
    # img_data.shape=[H,W]
    edge_value = (img_data[0][0] + img_data[0][-1] + img_data[-1][0] + img_data[-1][-1]) / 4
    if img_data.shape[0] % 2 == 0 and img_data.shape[1] % 2 == 0:
        padded_H = (target_size - img_data.shape[0]) // 2
        padded_W = (target_size - img_data.shape[1]) // 2
        img_data = np.lib.pad(img_data, ((padded_H, padded_H), (padded_W, padded_W)), mode="constant",
                              constant_values=edge_value)
    elif img_data.shape[0] % 2 == 1 and img_data.shape[1] % 2 == 0:
        padded_H = (target_size - img_data.shape[0]) // 2
        padded_W = (target_size - img_data.shape[1]) // 2
        img_data = np.lib.pad(img_data, ((padded_H, padded_H + 1), (padded_W, padded_W)), mode="constant",
                              constant_values=edge_value)
    elif img_data.shape[0] % 2 == 0 and img_data.shape[1] % 2 == 1:
        padded_H = (target_size - img_data.shape[0]) // 2
        padded_W = (target_size - img_data.shape[1]) // 2
        img_data = np.lib.pad(img_data, ((padded_H, padded_H), (padded_W, padded_W + 1)), mode="constant",
                              constant_values=edge_value)
    elif img_data.shape[0] % 2 == 1 and img_data.shape[1] % 2 == 1:
        padded_H = (target_size - img_data.shape[0]) // 2
        padded_W = (target_size - img_data.shape[1]) // 2
        img_data = np.lib.pad(img_data, ((padded_H, padded_H + 1), (padded_W, padded_W + 1)),
                              mode="constant", constant_values=edge_value)
    return img_data


def zeropadding3D(img_data, target_size):
    # img_data.shape=[C,H,W]
    if img_data.shape[1] % 2 == 0 and img_data.shape[2] % 2 == 0:
        padded_H = (target_size - img_data.shape[1]) // 2
        padded_W = (target_size - img_data.shape[2]) // 2
        img_data = np.lib.pad(img_data, ((0, 0), (padded_H, padded_H), (padded_W, padded_W)), mode="constant",
                              constant_values=0)
    elif img_data.shape[1] % 2 == 1 and img_data.shape[2] % 2 == 0:
        padded_H = (target_size - img_data.shape[1]) // 2
        padded_W = (target_size - img_data.shape[1]) // 2
        img_data = np.lib.pad(img_data, ((0, 0), (padded_H, padded_H + 1), (padded_W, padded_W)), mode="constant",
                              constant_values=0)
    elif img_data.shape[1] % 2 == 0 and img_data.shape[2] % 2 == 1:
        padded_H = (target_size - img_data.shape[1]) // 2
        padded_W = (target_size - img_data.shape[2]) // 2
        img_data = np.lib.pad(img_data, ((0, 0), (padded_H, padded_H), (padded_W, padded_W + 1)), mode="constant",
                              constant_values=0)
    elif img_data.shape[1] % 2 == 1 and img_data.shape[2] % 2 == 1:
        padded_H = (target_size - img_data.shape[1]) // 2
        padded_W = (target_size - img_data.shape[2]) // 2
        img_data = np.lib.pad(img_data, ((0, 0), (padded_H, padded_H + 1), (padded_W, padded_W + 1)),
                              mode="constant", constant_values=0)
    return img_data


def crop_center2D(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def crop_center3D(img, cropx, cropy):
    # img.shape=[C,H,W]
    _, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty + cropy, startx:startx + cropx]


def DataPreprocessing(img_data, zeropadding=False, target_size=512, center_crop2D=True, crop_size=280):
    """
    Perform some preprocessing to images such as zero padding, crop or new_normalization_on_GE.
    :param img_data: 
    :param zeropadding: 
    :param target_size: 
    :param center_crop2D: 
    :param crop_size: 
    :return: 
    """
    # Zero padding if "zeropadding=True"
    if zeropadding:
        img_data = zeropadding2D(img_data, target_size=target_size)
    # Center crop if "center_crop2D=True"
    if center_crop2D:
        img_data = crop_center2D(img_data, crop_size, crop_size)

    return img_data


if __name__ == "__main__":
    # test
    modals_dir1, modals_dir2 = get_selected_path()
    print(len(modals_dir1), len(modals_dir2))
    # print(modals_dir2)

    # x_resolution_all, x_y_resolution_all, z_resolution_all = resolution_statistics(get_resolution(modals_dir2))
    # resolution_hist(x_resolution_all)
    GE_2dData_list, Philips_2dData_list = store2dData2list()
    print(len(GE_2dData_list), len(Philips_2dData_list))
