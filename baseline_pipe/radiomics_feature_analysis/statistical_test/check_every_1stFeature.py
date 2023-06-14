""""
documentation describe: 
check every 1st order feature's all values
"""""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def check_func(csv_path):
    # get the dataframe
    df = pd.read_csv(csv_path, index_col=None)
    print(df.shape)
    CaseID = df.loc[:, "CaseID"].tolist()  # 按顺序存放着所有的CaseID

    feature_names = df.columns[1:].tolist()  # col_index=0 为 CaseID 列，所以从 col_index=1 开始取
    print("总共%d个一阶特征，如下面列表中所示：" % len(feature_names))
    print(feature_names)

    print("\n******************************循环开始******************************")
    for col_id in range(1, 19):  # col_index=0 为 CaseID 列，所以从 col_index=1 开始取，取到最后一列 col_index=19
        print("当前的一阶特征为%s" % feature_names[col_id - 1])
        # current feature's values
        current_feature_values = df.loc[:, feature_names[col_id - 1]].tolist()
        print("对应的数值总共有%d个，如下面列表所示" % len(current_feature_values))
        print(current_feature_values)
        print("-" * 80)

        # 我们的目的是绘制当前一阶特征的折线图，横坐标是CaseID，纵坐标是对应的一阶特征值
        plt.plot(current_feature_values)
        plt.title("Feature Name: %s" % feature_names[col_id - 1][40:])
        plt.xlabel("CaseID", fontsize=12)
        plt.ylabel("Values of the Current 1st Order Feature", fontsize=12)
        plt.savefig(feature_names[col_id - 1][40:] + ".png")
        plt.show()


if __name__ == "__main__":
    check_func(
        csv_path=r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\baseline_pipe\radiomics_feature_analysis\first_order_feature_test\new_csv\cervix_roi\Philips_ori_image_type_original(2).csv")
