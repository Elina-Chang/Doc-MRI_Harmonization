""""
Perform histogram matching on the original GE images and the PhilipsStyle GE images;
To get our ideal translated GE images and save them.
"""""
import SimpleITK as sitk
import glob
import os
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

from Tools.RootPath2CaseList import RootPath2CaseList
from Tools.Nii2Npy import Nii2Npy
from Tools.normalization1.normalization_after_clip import normalization_after_clip


def show_slice(slice):
    plt.imshow(slice, cmap="gray")
    plt.show()


def save_matching_results(root_path_ori_GE, root_path_PhilipsStyle_GE, save_root_path):
    """
    # save the matching results into every belong case
    :param root_path_ori_GE: the dir containg all the original GE cases
    :param root_path_PhilipsStyle_GE: the dir containg all the PhilipsStyle GE cases
    :param save_root_path:
    :return:
    """
    # go through every case of original GE and PhilipsStyle GE dir
    case_abs_path_ori_GE = RootPath2CaseList(root_path_ori_GE)  # return a list containing all the abs path
    case_abs_path_PhilipsStyle_GE = RootPath2CaseList(root_path_PhilipsStyle_GE)
    for path_ori_GE, path_PhilipsStyle_GE in zip(case_abs_path_ori_GE, case_abs_path_PhilipsStyle_GE):
        # get the ori GE images per each case
        ori_GE_img,ori_GE_arr = Nii2Npy(os.path.join(path_ori_GE, "t2sag.nii"))
        print(ori_GE_arr.min(), ori_GE_arr.max())
        # show_slice(ori_GE_arr[7])

        # get the PhilipsStyle GE images per each case
        PhilipsStyle_GE_img,PhilipsStyle_GE_arr = Nii2Npy(os.path.join(path_PhilipsStyle_GE, "t2sag_PhilipsStyle.nii.gz"))
        print(PhilipsStyle_GE_arr.min(), PhilipsStyle_GE_arr.max())
        # show_slice(PhilipsStyle_GE_arr[7])

        # perform the histogram matching, and get matched data(can perform on 3D data)
        ori_GE_arr = (4095 * (ori_GE_arr - ori_GE_arr.min()) / (ori_GE_arr.max() - ori_GE_arr.min())).astype(
            np.uint16)  # to [0,1].
        PhilipsStyle_GE_arr = (4095 * (PhilipsStyle_GE_arr - PhilipsStyle_GE_arr.min()) / (
                PhilipsStyle_GE_arr.max() - PhilipsStyle_GE_arr.min())).astype(np.uint16)
        matched_GE_arr = match_histograms(ori_GE_arr, PhilipsStyle_GE_arr, multichannel=True)
        # Norm1
        clip_value, matched_GE_arr = normalization_after_clip(matched_GE_arr)
        print(matched_GE_arr.min(), matched_GE_arr.max())
        print("-" * 60)
        # show_slice(matched_GE_arr[7])

        # save matched data to .nii.gz form(there wil be with the ori spacing info.)
        matched_GE_img = sitk.GetImageFromArray(matched_GE_arr)  # sitk.GetImageFromArray
        matched_GE_img.CopyInformation(ori_GE_img)  # image.CopyInformation
        save_path = os.path.join(save_root_path, os.path.basename(path_ori_GE))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sitk.WriteImage(matched_GE_img, os.path.join(save_path, "t2sag_matched.nii.gz"))


if __name__ == "__main__":
    save_matching_results(
        root_path_ori_GE=r"G:\PhD\Data_renji\Data_3T_Resampled_Norm1\SIEMENS_3T_Resampled_Norm1",
        root_path_PhilipsStyle_GE=r"G:\PhD\StyleTransferPredictions\newline_pipe_5\39SIEMENSpredictions",
        save_root_path=r"G:\PhD\Data_renji\Data_forFeatureExtractor\39SIEMENS")
# One Question: Why the coming out matched_GE_img are so gray compared to the ori_GE_img and the PhilipsStyle_GE_img
