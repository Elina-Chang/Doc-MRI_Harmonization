import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
from skimage.exposure import match_histograms


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
        root_path=r"E:\PhD\PycharmProjects\my_project\MRI_Harmonization\baseline_pipe"
                  r"\radiomics_feature_analysis\data_for_feature_extractor",
        ori_GE_dir=r"GE_ori_data", PhilipsStyle_GE_dir=r"PhilipsStyle_data",
        only_3T=True, csv_path_1_5T_info=r"E:\PhD\Data_renji\GE_data_info.csv",
        only_healthy=False, csv_path_abnormal_case=r"E:\PhD\Data_renji\Abnormal_case.xlsx",
        data_name=r"t2_sag_forFeatureExtractor.nii",
):
    case_list = os.listdir(os.path.join(root_path, ori_GE_dir))

    # eliminate 1.5T data if only_3T=True
    if only_3T:
        case_1_5T_list = get_1_5T_case(csv_path=csv_path_1_5T_info)
        case_list = list(set(case_list).difference(set(case_1_5T_list)))

    # eliminate abnormal case if only_healthy=True
    if only_healthy:
        GE_abnormal_case, _ = get_abnormal_case(csv_path=csv_path_abnormal_case)  # 请注意，这里是以集合形式存储
        case_list = list(set(case_list).difference(GE_abnormal_case))

    case_list = sorted(case_list)
    ori_data_path = [os.path.join(root_path, ori_GE_dir, case, data_name) for case in case_list]
    generated_data_path = [os.path.join(root_path, PhilipsStyle_GE_dir, case, data_name) for case in case_list]
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


def plt_hist_matching(ori_2Ddata_list, generated_2Ddata_list):
    img_id = 456#697
    for img_real, img_fake in zip(ori_2Ddata_list[img_id:], generated_2Ddata_list[img_id:]):
        img_real = (4095 * (img_real - img_real.min()) / (img_real.max() - img_real.min())).astype(np.uint16)
        img_fake = (4095 * (img_fake - img_fake.min()) / (img_fake.max() - img_fake.min())).astype(np.uint16)
        matched = match_histograms(img_real, img_fake)
        # plt.suptitle(img_id, fontsize=16)
        plt.subplot(2, 4, 1)
        plt.imshow(img_real, cmap="gray")
        plt.title("GE MRI")
        plt.subplot(2, 4, 2)
        plt.imshow(img_fake, cmap="gray")
        plt.title("PhilipsStyle MRI")
        plt.subplot(2, 4, 3)
        plt.imshow(matched, cmap="gray")
        plt.title("Matched GE MRI")
        plt.subplot(2, 4, 4)
        plt.imshow(matched - img_real, cmap="gray")
        plt.title("Error Image")
        plt.subplot(2, 4, 5)
        plt.hist(img_real.flatten(), bins=99)
        plt.title("GE MRI's Histogram")
        plt.subplot(2, 4, 6)
        plt.hist(img_fake.flatten(), bins=99)
        plt.title("PhilipsStyle MRI's Histogram")
        plt.subplot(2, 4, 7)
        plt.hist(matched.flatten(), bins=99)
        plt.title("Matched GE MRI's Histogram")
        plt.subplot(2, 4, 8)
        plt.hist((matched - img_real).flatten(), bins=99)
        plt.title("Error Image's Histogram")
        # plt.savefig(r"E:\PhD\Data_renji\Histogram Matching.svg", format="svg", dpi=600)
        plt.show()
        img_id += 1


if __name__ == '__main__':
    ori_2Ddata_list, generated_2Ddata_list = store2dData2list()
    print(len(ori_2Ddata_list), len(generated_2Ddata_list))
    plt_hist_matching(ori_2Ddata_list, generated_2Ddata_list)
