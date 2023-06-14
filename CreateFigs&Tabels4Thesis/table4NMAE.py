import os
import numpy as np
import pandas as pd

from Tools.RootPath2CaseList import RootPath2CaseList
from Tools.ShowData import ShowDataFromNpy
from Tools.Nii2Npy import Nii2Npy
from Tools.Ret1_5tCase import Ret1_5tCase
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sklearn.metrics import mean_absolute_error

root_path_ori = r"E:\PhD\Data_renji\Data_forFeatureExtractor\41GE"
root_path_trans = r"E:\PhD\Data_renji\Data_forFeatureExtractor_\41GE"
cases_ori = RootPath2CaseList(root_path_ori)
cases_trans = RootPath2CaseList(root_path_trans)

nmae_list = []
for case_ori, case_trans in zip(cases_ori, cases_trans):
    ori_nii = os.path.join(case_ori, "t2sag_ori.nii")
    harmonized_nii = os.path.join(case_trans, "t2sag_newline_pipe_1.nii.gz")
    _, ori_arr = Nii2Npy(ori_nii)
    _, harmonized_arr = Nii2Npy(harmonized_nii)
    # # check img
    # ShowDataFromNpy(ori_arr)
    # ShowDataFromNpy(harmonized_arr)

    arr_len = len(ori_arr)
    for i in np.arange(arr_len):
        ori_slice = ori_arr[i]
        harmonized_slice = harmonized_arr[i]
        nmae = mean_absolute_error(ori_slice, harmonized_slice) / np.mean(ori_slice)
        nmae_list.append(nmae)
df = pd.DataFrame({"nmae": nmae_list}, index=None)
print(df.describe())
