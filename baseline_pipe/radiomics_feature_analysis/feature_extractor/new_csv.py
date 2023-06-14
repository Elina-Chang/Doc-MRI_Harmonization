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


def get_small_FOV_data(PHILIPS_info_csv_path):
    PHILIPS_data_info = pd.read_excel(PHILIPS_info_csv_path, index_col=False)
    cases_list = PHILIPS_data_info.iloc[:, 0].tolist()
    return cases_list


def delete_cases_and_save(src_csv_path, CaseID_list, dst_csv_path):
    df = pd.read_csv(src_csv_path, index_col="CaseID")
    df.drop(CaseID_list).to_csv(dst_csv_path, index="CaseID")


if __name__ == "__main__":
    GE_cases_list = get_1_5T_data(GE_info_csv_path=r"E:\PhD\Data_renji\GE_data_info.csv")
    PHILIPS_cases_list = get_small_FOV_data(PHILIPS_info_csv_path=r"E:\PhD\Data_renji\Philips_data_info.xls")
    delete_cases_and_save(src_csv_path=r"E:\PhD\Radiomics\MyRadiomics\GE_PHILIPS_Classify_BaselinePipe"
                                       r"\cervix_roi\GE65+PHILIPS89\OriCSVs"
                                       r"\GE_styled_image_type_original(18+73).csv",
                          CaseID_list=GE_cases_list,
                          dst_csv_path=r"E:\PhD\Radiomics\MyRadiomics\GE_PHILIPS_Classify_BaselinePipe"
                                       r"\cervix_roi\GE41+PHILIPS75\OriCSVs"
                                       r"\GE_styled_image_type_original(18+73).csv")
