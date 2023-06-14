import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
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


def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf


def plt_hist_matching(ori_2Ddata_list, generated_2Ddata_list, img_id=656):
    for img_real, img_fake in zip(ori_2Ddata_list[img_id:], generated_2Ddata_list[img_id:]):
        img_real = (4095 * (img_real - img_real.min()) / (img_real.max() - img_real.min())).astype(np.uint16)
        img_fake = (4095 * (img_fake - img_fake.min()) / (img_fake.max() - img_fake.min())).astype(np.uint16)
        matched = match_histograms(img_real, img_fake)

        x1, y1 = ecdf(img_real.ravel())
        x2, y2 = ecdf(img_fake.ravel())
        x3, y3 = ecdf(matched.ravel())

        fig = plt.figure()
        gs = plt.GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()

        ax1.imshow(img_real, cmap=plt.cm.gray)
        ax1.set_title('Source')
        ax2.imshow(img_fake, cmap=plt.cm.gray)
        ax2.set_title('Reference')
        ax3.imshow(matched, cmap=plt.cm.gray)
        ax3.set_title('Matched')

        ax4.plot(x1, y1 * 100, '-r', lw=3, label='Source')
        ax4.plot(x2, y2 * 100, '-k', lw=3, label='Reference')
        ax4.plot(x3, y3 * 100, '--g', lw=3, label='Matched')
        ax4.set_xlim(x1[0], x1[-1])
        ax4.set_xlabel('Pixel value')
        ax4.set_ylabel('Cumulative %')
        ax4.legend(loc=5)

        # y1_ = np.zeros_like(y1)
        # ax4.fill_between(x1, y1, y1_, color='C0', alpha=0.2)
        # plt.savefig(r"Histogram Matching_{}.svg".format(img_id), format="svg", dpi=600)
        plt.show()


def plt_error_image(ori_2Ddata_list, generated_2Ddata_list):
    img_id = 656
    for img_real, img_fake in zip(ori_2Ddata_list[img_id:], generated_2Ddata_list[img_id:]):
        img_real = (4095 * (img_real - img_real.min()) / (img_real.max() - img_real.min())).astype(np.uint16)
        img_fake = (4095 * (img_fake - img_fake.min()) / (img_fake.max() - img_fake.min())).astype(np.uint16)
        matched = match_histograms(img_real, img_fake)
        error_img = (matched - img_real)
        error_img = (4095 * (error_img - error_img.min()) / (error_img.max() - error_img.min())).astype(np.uint16)

        fig = plt.figure()
        gs = plt.GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()

        ax1.imshow(img_real, cmap=plt.cm.gray)
        ax1.set_title('Source')
        ax2.imshow(matched, cmap=plt.cm.gray)
        ax2.set_title('Matched')
        ax3.imshow(error_img, cmap=plt.cm.gray)
        ax3.set_title('Difference')
        ax4.hist(img_real.flatten(), bins=99)
        ax4.set_title("Source's histogram")
        ax5.hist(matched.flatten(), bins=99)
        ax5.set_title("Matched's histogram")
        ax6.hist(error_img.flatten(), bins=99)
        ax6.set_title("Difference's histogram")
        plt.show()


if __name__ == "__main__":
    ori_2Ddata_list, generated_2Ddata_list = store2dData2list()
    print(len(ori_2Ddata_list), len(generated_2Ddata_list))
    # plt_hist_matching(ori_2Ddata_list, generated_2Ddata_list)
    plt_error_image(ori_2Ddata_list, generated_2Ddata_list)
