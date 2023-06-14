import glob
import os


def RootPath2CaseList(root_path):
    case_list = glob.glob(root_path + "\*")
    case_list = [case for case in case_list if os.path.isdir(case)]
    return case_list


if __name__ == "__main__":
    case_list = RootPath2CaseList(r"E:\PhD\Data_multivendor\SIEMENS")
    print(case_list)
