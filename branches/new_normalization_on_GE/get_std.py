import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def get_std(input_csv_path):
    df = pd.read_csv(input_csv_path, index_col=None)
    features_name_list = df.describe().columns.tolist()

    # # 使用min-max归一化数据
    # ss = MinMaxScaler()
    # df[features_name_list] = ss.fit_transform(df[features_name_list])

    # # 使用z-score标准化数据
    # ss = StandardScaler()
    # df[features_name_list] = ss.fit_transform(df[features_name_list])

    # # 使用RobustScaler标准化
    # ss = RobustScaler()
    # df[features_name_list] = ss.fit_transform(df[features_name_list])

    # Normalize to unit
    ss = Normalizer()
    df[features_name_list] = ss.fit_transform(df[features_name_list])


    features_name_list = [i[44:] for i in features_name_list]
    std_list = df.describe().loc["std"].values.tolist()
    return features_name_list, std_list


def plot_std_scatter(input_csv_path):
    features_name_list, std_list = get_std(input_csv_path=input_csv_path)
    # std_list[2] = None
    # std_list[-3] = None

    plt.scatter(np.arange(0, 18, 1), std_list, c="blue")
    plt.xticks(np.arange(0, 18, 1))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title("GE组内(41例)特征(18个)std分数分布情况", fontsize=16)
    plt.xlabel("特征编号", fontsize=14)
    plt.ylabel("std分数", fontsize=14)
    plt.show()


def record_std(input_csv_path, output_path, normalization="Normalization0_std"):
    features_name_list, std_list = get_std(input_csv_path=input_csv_path)
    data = {"FeatureName": features_name_list, normalization: std_list}
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)


# def plot_multi_std_scatter(input_csv0_path, input_csv1_path):
#     df0 = pd.read_excel(input_csv0_path, index_col=None)
#     df1 = pd.read_excel(input_csv1_path, index_col=None)
#
#     std_list0 = df0.iloc[:, -1].values.tolist()
#     std_list0[2] = None
#     std_list0[-3] = None
#
#     std_list1 = df1.iloc[:, -1].values.tolist()
#     std_list1[2] = None
#     std_list1[-3] = None
#
#     plt.scatter(np.arange(0, 18, 1), std_list0)
#     plt.scatter(np.arange(0, 18, 1), std_list1)
#     plt.legend(['Min-max', 'Normalization0'])
#     plt.xticks(np.arange(0, 18, 1))
#     plt.xlim(left=0)
#     plt.ylim(bottom=0)
#     plt.title("GE组内(41例)特征(18个)std分数分布情况", fontsize=16)
#     plt.xlabel("特征编号", fontsize=14)
#     plt.ylabel("std分数", fontsize=14)
#     plt.show()


def plot_multi_std_scatter(*kwargs_csv_path):
    for csv_path in kwargs_csv_path:
        df = pd.read_excel(csv_path, index_col=None)
        std_list = df.iloc[:, -1].values.tolist()
        # std_list[2] = None
        # std_list[-3] = None
        plt.scatter(np.arange(0, 18, 1), std_list)
    plt.legend(["Min-max", "Normalization0", "Normalization1"])
    plt.xticks(np.arange(0, 18, 1))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title("GE组内(41例)特征(18个)std分数分布情况", fontsize=16)
    plt.xlabel("特征编号", fontsize=14)
    plt.ylabel("std分数", fontsize=14)
    plt.show()


def main():
    csv0_path = r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\branches" \
                r"\GE_ori_first_order_feature_u_test\GE_3T_ori_image_type_original(18).csv"
    excel0_path = r"G:\Presentations\一阶特征分析实验\组内一阶特征分析\降低组内特征std" \
                  r"\df_norm_min-maxGE组内std(womb)_3T数据(41例+18个特征).xlsx"
    record_std(input_csv_path=csv0_path, output_path=excel0_path)
    plot_std_scatter(input_csv_path=csv0_path)

    csv1_path = r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\branches" \
                r"\new_normalization_on_GE\GE_3T_normalization0_image_type_original(18).csv"
    excel1_path = r"G:\Presentations\一阶特征分析实验\组内一阶特征分析\降低组内特征std" \
                  r"\df_norm_normalization0GE组内std(womb)_3T数据(41例+18个特征).xlsx"
    record_std(input_csv_path=csv1_path, output_path=excel1_path)
    plot_std_scatter(input_csv_path=csv1_path)

    csv2_path = r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\branches" \
                r"\new_normalization_on_GE\GE_3T_normalization1_image_type_original(18).csv"
    excel2_path = r"G:\Presentations\一阶特征分析实验\组内一阶特征分析\降低组内特征std" \
                  r"\df_norm_normalization1GE组内std(womb)_3T数据(41例+18个特征).xlsx"
    record_std(input_csv_path=csv2_path, output_path=excel2_path)
    plot_std_scatter(input_csv_path=csv2_path)


if __name__ == "__main__":
    main()
    excel0_path = r"G:\Presentations\一阶特征分析实验\组内一阶特征分析\降低组内特征std\df_norm_min-maxGE组内std(womb)_3T数据(41例+18个特征).xlsx"
    excel1_path = r"G:\Presentations\一阶特征分析实验\组内一阶特征分析\降低组内特征std\df_norm_normalization0GE组内std(womb)_3T数据(41例+18个特征).xlsx"
    excel2_path = r"G:\Presentations\一阶特征分析实验\组内一阶特征分析\降低组内特征std\df_norm_normalization1GE组内std(womb)_3T数据(41例+18个特征).xlsx"
    plot_multi_std_scatter(excel0_path, excel1_path, excel2_path)
