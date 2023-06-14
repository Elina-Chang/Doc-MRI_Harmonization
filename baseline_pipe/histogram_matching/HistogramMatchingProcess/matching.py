import numpy as np
from Tools.Nii2Npy import Nii2Npy
from Tools.RootPath2CaseList import RootPath2CaseList
import SimpleITK as sitk

# from Tools.ShowData import ShowDataFromNii
# ShowDataFromNii(r"E:\PhD\Data_renji\Data_forFeatureExtractor\41GE\CAO_HAI_XIA_RJN1015954\t2sag_newmatched.nii.gz")
# image = sitk.ReadImage(r"E:\PhD\Data_renji\Data_forFeatureExtractor\41GE\CAO_HAI_XIA_RJN1015954\t2sag_newmatched.nii.gz")
# data = sitk.GetArrayFromImage(image)
# print(data.dtype,data.min(),data.max())


def MatchedUsingLUT(src_arr, LUT):
    rescaled_src_arr = (4095 * (src_arr - src_arr.min()) / (src_arr.max() - src_arr.min())).astype(np.uint16)

    for i in range(rescaled_src_arr.shape[0]):
        for j in range(rescaled_src_arr.shape[1]):
            for k in range(rescaled_src_arr.shape[2]):
                rescaled_src_arr[i, j, k] = LUT[rescaled_src_arr[i, j, k]]
    return rescaled_src_arr


case_root_path = r"E:\PhD\Data_renji\Data_forFeatureExtractor\41GE"
LUT_root_path = r"/baseline_pipe/histogram_matching/LUTresults"
case_path_list = RootPath2CaseList(case_root_path)[:]
LUT_path_list = RootPath2CaseList(LUT_root_path)[:]
for case_path, LUT_path in zip(case_path_list, LUT_path_list):
    src_img, src_arr = Nii2Npy(case_path + "\\" + "t2sag_ori.nii")
    print(src_arr.shape,src_arr.dtype)
    LUT = list(np.load(LUT_path + "\\" + "FinalLUT.npy").astype(np.uint16))
    matched_arr = MatchedUsingLUT(src_arr, LUT)
    matched_arr = (matched_arr - matched_arr.min()) / (matched_arr.max() - matched_arr.min()).astype(np.float32)
    print(matched_arr.shape,matched_arr.dtype)
    matched_img = sitk.GetImageFromArray(matched_arr)
    matched_img.CopyInformation(src_img)
    sitk.WriteImage(matched_img, case_path + "\\" + "t2sag_newmatched.nii.gz")
