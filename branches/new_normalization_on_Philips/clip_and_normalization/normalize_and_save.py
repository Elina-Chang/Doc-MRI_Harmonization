import os
import numpy as np
import SimpleITK as sitk

from branches.new_normalization_on_Philips.clip_and_normalization.clip.generate_clip_value import generate_clip_value
from baseline_pipe.radiomics_feature_analysis.data_for_feature_extractor.check_and_store_data.SaveModel import \
    SaveNumpyToImageByRef


def get_img_path_list(root_path=r"G:\PhD\Data_renji\Philips_resampled"):
    cases_list = os.listdir(root_path)
    img_path_list = [os.path.join(root_path, case_name, "t2_sag_Resample.nii") for case_name in cases_list]
    return img_path_list


def get_img_data(img_path):
    img = sitk.ReadImage(img_path)
    img_data = sitk.GetArrayFromImage(img)
    # img_data = np.transpose(img_data, (1, 2, 0))
    # img_data = np.flipud(img_data)
    # img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    return img, img_data


def normalization_after_clip(data3D):
    clip_value = generate_clip_value(data3D)
    data3D[data3D < clip_value["bottom"]] = clip_value["bottom"]
    data3D[data3D > clip_value["top"]] = clip_value["top"]
    data3D = (data3D - data3D.min()) / (data3D.max() - data3D.min())
    return clip_value, data3D


def main(store_root_path):
    img_path_list = get_img_path_list()
    for img_path in img_path_list:
        print(img_path)
        img, img_data = get_img_data(img_path)
        clip_value, normalized_data = normalization_after_clip(img_data)

        normalized_data = np.transpose(normalized_data, [1, 2, 0])
        normalized_data = np.flipud(normalized_data)

        # write the normalized_data
        store_path = os.path.join(store_root_path, img_path.split(sep="\\")[4])
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        SaveNumpyToImageByRef(os.path.join(store_path, "t2sag_resampled_normalization1.nii"), normalized_data, img)


def temp(store_root_path):
    img_path_list = get_img_path_list()[-1:]
    for img_path in img_path_list:
        print(img_path)
        img, img_data = get_img_data(img_path)
        clip_value, normalized_data = normalization_after_clip(img_data)
        print(clip_value)

        normalized_data = np.transpose(normalized_data, [1, 2, 0])
        normalized_data = np.flipud(normalized_data)

        # write the normalized_data
        store_path = os.path.join(store_root_path, img_path.split(sep="\\")[4])
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        SaveNumpyToImageByRef(os.path.join(store_path, "t2sag_resampled_normalization1.nii"), normalized_data, img)


if __name__ == "__main__":
    store_root_path = r"G:\PhD\Data_renji\Philips_resampled"
    # temp(store_root_path)
    main(store_root_path)
