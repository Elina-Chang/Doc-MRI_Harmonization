import os
import numpy as np
import SimpleITK as sitk

from baseline_pipe.radiomics_feature_analysis.data_for_feature_extractor.check_and_store_data.SaveModel import \
    SaveNumpyToImageByRef
from Tools.RootPath2CaseList import RootPath2CaseList
from Tools.Nii2Npy import Nii2Npy
from Tools.normalization1.normalization_after_clip import normalization_after_clip


def get_img_path_list(root_path=r"G:\PhD\Data_renji\SIEMENS_3T_Resampled"):
    cases = RootPath2CaseList(root_path)
    img_path_list = [os.path.join(case, "t2sag.nii.gz") for case in cases]
    return img_path_list


def main(store_root_path):
    img_path_list = get_img_path_list()
    for img_path in img_path_list:
        print(img_path)
        img, img_data = Nii2Npy(img_path)
        clip_value, normalized_data = normalization_after_clip(img_data)

        normalized_data = np.transpose(normalized_data, [1, 2, 0])
        normalized_data = np.flipud(normalized_data)

        # write the normalized_data
        store_path = os.path.join(store_root_path, img_path.split(sep="\\")[4])
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        SaveNumpyToImageByRef(os.path.join(store_path, "t2sag.nii.gz"), normalized_data, img)


def temp(store_root_path):
    img_path_list = get_img_path_list()[-1:]
    for img_path in img_path_list:
        print(img_path)
        img, img_data = Nii2Npy(img_path)
        clip_value, normalized_data = normalization_after_clip(img_data)
        print(clip_value)

        normalized_data = np.transpose(normalized_data, [1, 2, 0])
        normalized_data = np.flipud(normalized_data)

        # write the normalized_data
        store_path = os.path.join(store_root_path, img_path.split(sep="\\")[4])
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        SaveNumpyToImageByRef(os.path.join(store_path, "t2sag.nii.gz"), normalized_data, img)


if __name__ == "__main__":
    store_root_path = r"G:\PhD\Data_renji\SIEMENS_3T_Resampled_Norm1"
    # temp(store_root_path)
    main(store_root_path)
