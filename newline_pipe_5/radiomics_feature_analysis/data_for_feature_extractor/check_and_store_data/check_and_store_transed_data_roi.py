import SimpleITK as sitk
import numpy as np
import glob
import os

from Tools.Visualization import Imshow3DArray
from Tools.SaveModel import SaveNumpyToImageByRef


def examine_and_store_data(img_root_path, roi_root_path, store_path):
    cases_path_img = glob.glob(img_root_path + "\*")
    cases_path_roi = glob.glob(roi_root_path + "\*")
    for case_img, case_roi in zip(cases_path_img, cases_path_roi):
        print(os.path.basename(case_img))

        img_path = glob.glob(case_img + "\*t2sag_proposed.nii.gz")[0]
        img = sitk.ReadImage(img_path)
        if 0.65 < img.GetSpacing()[0] < 0.85:
            img_data = sitk.GetArrayFromImage(img)
            img_data = np.transpose(img_data, [1, 2, 0])
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            # print(img_data.shape, img_data.dtype, img_data.min(), img_data.max())
            # img_data = np.flipud(img_data)

        roi_path = glob.glob(case_roi + "\*cervix_roi.nii.gz")[0]
        roi = sitk.ReadImage(roi_path)
        if 0.65 < roi.GetSpacing()[0] < 0.85:
            roi_data = sitk.GetArrayFromImage(roi)
            roi_data = np.transpose(roi_data, [1, 2, 0])
            # print(roi_data.shape, roi_data.dtype, roi_data.min(), roi_data.max())
            # roi_data = np.flipud(roi_data)

        Imshow3DArray(data=img_data, roi=roi_data)

        # # store img_data and roi_data
        # case_store_path = os.path.join(store_path, os.path.basename(case_img))
        # if not os.path.exists(case_store_path):
        #     os.makedirs(case_store_path)
        # SaveNumpyToImageByRef(os.path.join(case_store_path, r"t2sag_cycleGAN.nii.gz"),
        #                       data=img_data, ref_image=img)
        # SaveNumpyToImageByRef(os.path.join(case_store_path, r"cervix_roi.nii.gz"),
        #                       data=roi_data, ref_image=roi)


if __name__ == "__main__":
    examine_and_store_data(img_root_path=r"G:\PhD\Data_renji\Data_forFeatureExtractor\GE_3T",
                           roi_root_path=r"G:\PhD\Data_renji\Data_forFeatureExtractor\GE_3T",
                           store_path=r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\vanilla_cycleGAN"
                                      r"\radiomics_feature_analysis\data_for_feature_extractor\PhilipsStyle_data")
