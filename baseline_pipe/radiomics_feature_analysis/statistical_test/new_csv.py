""""
delete 1.5T cases in first order feature csv.
"""""
import numpy as np
import os
import pandas as pd


def get_1_5T_data(GE_info_csv_path):
    # 读取包含GE数据扫描信息的.csv文件，并存储1.5T的cases
    GE_data_info = pd.read_csv(GE_info_csv_path, index_col=False)
    cases_list = []  # 用于存放1.5T的cases
    for row_id in range(GE_data_info.shape[0]):
        row = list(GE_data_info.iloc[row_id, :])
        # 筛选cases
        if "b'1.5T" in row[1]:
            cases_list.append(row[0])
    return cases_list


def delete_1_5T_row_and_save(first_order_csv_path, CaseID_list, store_name):
    first_order_df = pd.read_csv(first_order_csv_path, index_col="CaseID")
    first_order_df.drop(CaseID_list).to_csv(store_name, index="CaseID")


if __name__ == "__main__":
    cases_list = get_1_5T_data(GE_info_csv_path=r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization"
                                                r"\baseline_pipe\GE_data_info.csv")
    delete_1_5T_row_and_save(first_order_csv_path=r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\baseline_pipe"
                                                  r"\radiomics_feature_analysis\feature_extractor"
                                                  r"\csv\womb_roi\GE_ori_image_type_original(18).csv",
                             CaseID_list=cases_list, store_name="GE_3T_ori_image_type_original(18).csv"
                             )
