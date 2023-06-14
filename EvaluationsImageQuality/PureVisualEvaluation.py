""""
Evalue the generated images with the original images by our eyes.
"""""
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd


def get_abnormal_case(csv_path=r".\Abnormal_case.xlsx"):
    """
    # Acquire the cases' names of the abnormal data
    :param csv_path: .csv file path for storing the abnormal info
    :return: (set)形式: {}
    """
    csv = pd.read_excel(csv_path, index_col=None)
    GE_abnormal_cases = set(csv.iloc[:, 0].dropna())
    Philips_abnormal_cases = set(csv.iloc[:, 1].dropna())

    return GE_abnormal_cases, Philips_abnormal_cases


def get_1_5T_case(csv_path=r".\GE_data_info.csv"):
    """
    # Acquire the cases' names of the 1.5T data
    :param csv_path: .csv file path for storing the 1.5T info
    :return:
    """
    # 读取包含扫描信息的.csv文件，得到1.5T的cases
    data_info = pd.read_csv(csv_path, index_col=False)
    case_1_5T = []  # 用于存放要剔除的cases
    for row_id in range(data_info.shape[0]):
        row = list(data_info.iloc[row_id, :])
        # 筛选cases
        if "b'1.5T" in row[1]:
            case_1_5T.append(row[0][:-8])

    return case_1_5T


def get_selected_data_path(
        root_dir=r"G:\PhD\PycharmProjects\my_project\CycleGAN_MRI\CycleGAN_MRI\all_data\GE_womb_t2sag",
        only_3T=False, csv_path_1_5T_info=r"G:\PhD\PycharmProjects\my_project\CycleGAN_MRI\CycleGAN_MRI"
                                         r"\all_statistics\GE_data_info.csv",
        only_healthy=True, csv_path_abnormal_case=r"G:\PhD\Data_renji\Abnormal_case.xlsx",
        original_data_name=r"t2_sag.nii",
        generated_data_name=r"t2_sag_PhilipsStyle_train_3'_partial_data_log01.nii",
):
    case_list = os.listdir(root_dir)

    # eliminate 1.5T data if only_3T=True
    if only_3T:
        case_1_5T_list = get_1_5T_case(csv_path=csv_path_1_5T_info)
        case_list = list(set(case_list).difference(set(case_1_5T_list)))

    # eliminate abnormal case if only_healthy=True
    if only_healthy:
        GE_abnormal_case, _ = get_abnormal_case(csv_path=csv_path_abnormal_case)  # 请注意，这里是以集合形式存储
        case_list = list(set(case_list).difference(GE_abnormal_case))

    case_list = sorted(case_list)
    ori_data_path = [os.path.join(root_dir, case, original_data_name) for case in case_list]
    generated_data_path = [os.path.join(root_dir, case, generated_data_name) for case in case_list]
    print(len(ori_data_path), len(generated_data_path))

    return ori_data_path, generated_data_path


def store3dData2list():
    ori_data_path, generated_data_path = get_selected_data_path()
    ori_3Ddata_list = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in ori_data_path]
    generated_3Ddata_list = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in generated_data_path]

    return ori_3Ddata_list, generated_3Ddata_list


def store2dData2list():
    ori_3Ddata_list, generated_3Ddata_list = store3dData2list()
    ori_2Ddata_temp_list = [[x3d[i] for i in range(x3d.shape[0])] for x3d in ori_3Ddata_list]
    generated_2Ddata_temp_list = [[x3d[i] for i in range(x3d.shape[0])] for x3d in generated_3Ddata_list]
    from functools import reduce
    ori_2Ddata_list = reduce(lambda x, y: x + y, ori_2Ddata_temp_list)
    generated_2Ddata_list = reduce(lambda x, y: x + y, generated_2Ddata_temp_list)

    return ori_2Ddata_list, generated_2Ddata_list


def evalue_images(ori_2Ddata_list, generated_2Ddata_list):
    """
    传参：形如[24,280,280]的数组存放所有2D数据
    :param ori_2Ddata_list:
    :param generated_2Ddata_list:
    :return:
    """
    img_id = 0
    for img_real, img_fake in zip(ori_2Ddata_list[img_id:], generated_2Ddata_list[img_id:]):
        from all_trains_and_involved.prepare_data import crop_center2D
        img_real = crop_center2D(img_real, 280, 280)
        img_fake = crop_center2D(img_fake, 280, 280)
        img_real = (4095 * (img_real - img_real.min()) / (img_real.max() - img_real.min())).astype(np.uint16)
        img_fake = (4095 * (img_fake - img_fake.min()) / (img_fake.max() - img_fake.min())).astype(np.uint16)

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(img_real, cmap="gray")
        plt.title("GE MRI")
        plt.subplot(2, 2, 3)
        plt.hist(img_real.flatten(), bins=199)
        plt.title("GE MRI's Histogram")
        plt.subplot(2, 2, 2)
        plt.imshow(img_fake, cmap="gray")
        plt.title("PhilipsStyle MRI")
        plt.subplot(2, 2, 4)
        plt.hist(img_fake.flatten(), bins=199)
        plt.title("PhilipsStyle MRI's Histogram")
        plt.suptitle(img_id, fontsize=16)
        plt.show()
        img_id += 1


if __name__ == '__main__':
    ori_2Ddata_list, generated_2Ddata_list = store2dData2list()
    print(len(ori_2Ddata_list), len(generated_2Ddata_list))
    # evalue_images(ori_2Ddata_list, generated_2Ddata_list)
