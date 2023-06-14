import os
import numpy as np
import pandas as pd

from Tools.RootPath2CaseList import RootPath2CaseList
from Tools.ShowData import ShowDataFromNpy
from Tools.Nii2Npy import Nii2Npy
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sklearn.metrics import mean_absolute_error

ori_root_path = r"E:\PhD\Data_renji\Data_3T_Resampled_Norm1\GE_3T_Resampled_Norm1"
vanicycle_root_path = r"E:\PhD\StyleTransferPredictions\vanilla_cycleGAN\41GEpredictions"

ori_cases = RootPath2CaseList(ori_root_path)
vani_cases = RootPath2CaseList(vanicycle_root_path)
nmae_list = []
psnr_list = []
mssim_list = []
for ori_case, vani_case in zip(ori_cases, vani_cases):
    ori_nii = os.path.join(ori_case, "t2sag.nii.gz")
    vani_nii = os.path.join(vani_case, "t2sag_PhilipsStyle.nii.gz")
    _, ori_arr = Nii2Npy(ori_nii)
    _, vani_arr = Nii2Npy(vani_nii)
    # vani_arr = np.transpose(np.flipud(np.transpose(vani_arr, [1, 2, 0])), [2, 0, 1])
    # # check img
    # ShowDataFromNpy(ori_arr)
    # ShowDataFromNpy(vani_arr)
    arr_len = len(ori_arr)
    for i in np.arange(arr_len):
        ori_slice = ori_arr[i]
        vani_slice = vani_arr[i]

        nmae = mean_absolute_error(ori_slice, vani_slice)
        psnr = peak_signal_noise_ratio(ori_slice, vani_slice)
        mssim = structural_similarity(ori_slice, vani_slice, gaussian_weights=True, sigma=1.5,
                                      use_sample_covariance=False)
        nmae_list.append(nmae)
        psnr_list.append(psnr)
        mssim_list.append(mssim)
df_psnr = pd.DataFrame({"psnr": psnr_list}, index=None)
df_mssim = pd.DataFrame({"mssim": mssim_list}, index=None)
df_nmae = pd.DataFrame({"nmae": nmae_list}, index=None)

print(df_psnr.describe(), df_mssim.describe(), df_nmae.describe())
