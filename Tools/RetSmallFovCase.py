import pandas as pd


def RetSmallFovCase(PHILIPS_info_csv_path):
    PHILIPS_data_info = pd.read_excel(PHILIPS_info_csv_path, index_col=False)
    cases_list = PHILIPS_data_info.iloc[:, 0].tolist()
    return cases_list
