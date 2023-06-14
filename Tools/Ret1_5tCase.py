import pandas as pd


def Ret1_5tCase(GE_info_csv_path):
    # 读取包含GE数据扫描信息的.csv文件，并存储1.5T的cases
    GE_data_info = pd.read_csv(GE_info_csv_path, index_col=False)
    cases_list = []  # 用于存放1.5T的cases
    for row_id in range(GE_data_info.shape[0]):
        row = list(GE_data_info.iloc[row_id, :])
        # 筛选cases
        if "b'1.5T" in row[1]:
            cases_list.append(row[0])
    return cases_list
