import scipy.stats as stats
import pandas as pd


def u_test(GE_csv_path, Philips_csv_path):
    GE_csv = pd.read_csv(GE_csv_path, index_col=False)  # 读取两个.csv文件
    Philips_csv = pd.read_csv(Philips_csv_path, index_col=False)

    GE_columns_list = list(GE_csv.columns)  # 取出所有的columns名称，并存放在list中
    Philips_columns_list = list(Philips_csv.columns)

    u_statistic_list = []  # 用于存放统计结果
    pVal_list = []
    feature_names = []  # 用于存放有显著性差异的特征名称
    statistic_list = []  # 存放有显著性特征的统计结果（包括特征名，p指和u检验值）
    for i in range(GE_csv.shape[1] - 1):  # 左开右闭，循环遍历所有特征，i=0,1,2,...
        print(i + 1)  # i=0列为CaseID，特征从（i+1）列开始
        GE_feature = list(GE_csv.iloc[:, i + 1])  # 取出（i+1）列的特征值，存放在list中
        Philips_feature = list(Philips_csv.iloc[:, i + 1])

        u_statistic, pVal = stats.mannwhitneyu(GE_feature, Philips_feature)
        # 存放所有的统计值
        u_statistic_list.append(u_statistic)  # 依次存放每个特征对应的 u 检验统计值
        pVal_list.append(pVal)  # 存放p值

        if pVal < 0.05:
            print(i + 1)
            if GE_columns_list[i + 1] == Philips_columns_list[i + 1]:
                feature_names.append(GE_columns_list[i + 1])
            else:
                feature_names.append(GE_columns_list[i + 1])
            print("u检验的结果：")
            print("u_statistic =", u_statistic, "   p_value =", pVal)
            print("具有显著性差异的特征是: ", GE_columns_list[i + 1])
            print("######################################################\n")
            statistic_list.append([GE_columns_list[i + 1][31:], pVal, u_statistic])
    print(len(u_statistic_list), len(pVal_list))
    print("有显著性的特征数为：", len(feature_names))
    print("有显著性的特征分别为：")
    for item in feature_names:
        print(item)

    print(statistic_list)
    statistic_df = pd.DataFrame(statistic_list, columns=["Feature name", "P value", "U statistic"])
    statistic_excel = statistic_df.to_excel(r".\ImageType=Wavelet\AllCases\new_u_test_results(GE_vs_Philips).xlsx",
                                            index=None)


if __name__ == "__main__":
    GE_csv_path = r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\newline_pipe_1" \
                  r"\radiomics_feature_analysis\first_order_feature_test\new_csv\PhilipsStyle_3T_image_type_original.csv"
    Philips_csv_path = r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\baseline_pipe" \
                       r"\radiomics_feature_analysis\first_order_feature_test\new_csv\Philips_ori_image_type_original.csv"
    u_test(GE_csv_path, Philips_csv_path)
