import scipy.stats as stats
import pandas as pd
import numpy as np
from random import shuffle


def seperate_GE_ori_csv(GE_ori_csv_path):
    """

    :return: two separate dataframe.
    """
    # read .csv
    df = pd.read_csv(GE_ori_csv_path, index_col=None)
    row_num = df.shape[0]
    row_id_list = np.arange(row_num).tolist()
    shuffle(row_id_list)
    print(row_id_list[:row_num // 2])
    print(row_id_list[row_num // 2:])
    df1 = df.iloc[row_id_list[:row_num // 2]]
    df2 = df.iloc[row_id_list[row_num // 2:]]

    return df1, df2


def u_test(GE_ori_csv_path):
    """

    :param ori_Philips_csv_path:
    :return:
    """

    df1, df2 = seperate_GE_ori_csv(GE_ori_csv_path)

    df1_columns_list = list(df1.columns)  # 取出所有的columns名称，并存放在list中
    df2_columns_list = list(df2.columns)

    u_statistic_list = []  # 用于存放统计结果
    pVal_list = []
    feature_names = []  # 用于存放有显著性差异的特征名称
    statistic_list = []  # 存放有显著性特征的统计结果（包括特征名，p指和u检验值）
    for i in range(df1.shape[1] - 1):  # 左开右闭，循环遍历所有特征，i=0,1,2,...
        print(i + 1)  # i=0列为CaseID，特征从（i+1）列开始
        df1_feature = list(df1.iloc[:, i + 1])  # 取出（i+1）列的特征值，存放在list中
        df2_feature = list(df2.iloc[:, i + 1])

        u_statistic, pVal = stats.mannwhitneyu(df1_feature, df2_feature)
        # 存放所有的统计值
        u_statistic_list.append(u_statistic)  # 依次存放每个特征对应的 u 检验统计值
        pVal_list.append(pVal)  # 存放p值

        if pVal < 0.05:
            print(i + 1)
            if df1_columns_list[i + 1] == df2_columns_list[i + 1]:
                feature_names.append(df1_columns_list[i + 1])
            else:
                feature_names.append(df1_columns_list[i + 1])
            print("u检验的结果：")
            print("u_statistic =", u_statistic, "   p_value =", pVal)
            print("具有显著性差异的特征是: ", df1_columns_list[i + 1])
            print("######################################################\n")
            statistic_list.append([df1_columns_list[i + 1][31:], pVal, u_statistic])
    print(len(u_statistic_list), len(pVal_list))
    print("有显著性的特征数为：", len(feature_names))
    print("有显著性的特征分别为：")
    for item in feature_names:
        print(item)

    print(statistic_list)
    statistic_df = pd.DataFrame(statistic_list, columns=["Feature name", "P value", "U statistic"])
    # statistic_excel = statistic_df.to_excel(r"E:\PhD\PycharmProjects\my_project\MRI_Harmonization\baseline_pipe"
    #                                         r"\radiomics_feature_analysis\statistical_test\u_test_results"
    #                                         r"\PhilipsStyle_vs_Philips_ImageType=Original.xlsx",
    #                                         index=None)


if __name__ == "__main__":
    GE_ori_csv_path = r".\GE_3T_ori_image_type_original(18).csv"
    u_test(GE_ori_csv_path)
