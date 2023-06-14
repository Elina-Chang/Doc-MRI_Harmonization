""""
Resample and rescale data, and check the img_data and roi_data and store them for later feature extraction.
"""""

import glob
import numpy as np
import SimpleITK as sitk
import os

from baseline_pipe.radiomics_feature_analysis.data_for_feature_extractor.check_and_store_data.Visualization import \
    Imshow3DArray
from baseline_pipe.radiomics_feature_analysis.data_for_feature_extractor.check_and_store_data.SaveModel import \
    SaveNumpyToImageByRef
from cyclegan_mri.Resampler import Resampler

resampler = Resampler()


def examine_and_store_data(root_path, store_path=None):
    """
    查看配好roi的图像，归一化后存储：
        params:
            root_path: 含有所有case的根目录
            store_path: 存放所有case的根目录
    """
    cases = glob.glob(root_path + "\*")  # 获得所有case的绝对路径
    # cases.remove(os.path.join(root_path, "output.txt"))
    for case in cases[:]:  # 遍历所有case的绝对路径
        print(case)
        nii_name_list = os.listdir(case)  # case里面包含的.nii的name

        # THE image
        img_path = glob.glob(case + "\*t2sag_newmatched.nii.gz")[0]
        img = sitk.ReadImage(img_path)
        # if img.GetSpacing()[0] < 0.75 or img.GetSpacing()[0] > 0.85:  # resample img_data
        #     resampled_img = resampler.ResizeSipmleITKImage(image=img_path, is_roi=False,
        #                                                    expected_resolution=[0.75, 0.75, -1])
        #     img_data = sitk.GetArrayFromImage(resampled_img)
        # else:
        #     print("ELSE~!")
        #     img_data = sitk.GetArrayFromImage(img)
        img_data = sitk.GetArrayFromImage(img)
        img_data = np.transpose(img_data, [1, 2, 0])  # Transpose: [C,H,W]——>[H,W,C]
        print(img_data.shape, img_data.dtype, img_data.min(), img_data.max())
        # img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min())).astype(np.float32)  # Normalization
        # print(img_data.shape, img_data.dtype, img_data.min(), img_data.max())
        # img_data = np.flipud(img_data)

        # THE roi
        roi_path = glob.glob(case + "\*cervix_roi.nii.gz")[0]
        roi = sitk.ReadImage(roi_path)
        # if roi.GetSpacing()[0] < 0.65 or roi.GetSpacing()[0] > 0.85:  # resample roi_data
        #     resampled_roi = resampler.ResizeSipmleITKImage(image=roi_path, is_roi=True,
        #                                                    expected_resolution=[0.7, 0.7, -1])
        #     roi_data = sitk.GetArrayFromImage(resampled_roi)
        # else:
        #     roi_data = sitk.GetArrayFromImage(roi)
        roi_data = sitk.GetArrayFromImage(roi)
        roi_data = np.transpose(roi_data, [1, 2, 0])  # [C,H,W]——>[H,W,C]       # Transpose
        print(roi_data.shape, roi_data.dtype, roi_data.min(), roi_data.max())
        # roi_data = np.flipud(roi_data)

        Imshow3DArray(img_data, roi=roi_data)

        # # store img_data and roi_data
        # case_store_path = os.path.join(store_path, os.path.basename(case))
        # if not os.path.exists(case_store_path):
        #     os.makedirs(case_store_path)
        # # store img_data
        # if img.GetSpacing()[0] < 0.65 or img.GetSpacing()[0] > 0.85:
        #     SaveNumpyToImageByRef(
        #         os.path.join(case_store_path, "t2sag_newline_pipe_1.nii.gz"),
        #         img_data, resampled_img)
        # else:
        #     SaveNumpyToImageByRef(
        #         os.path.join(case_store_path, "t2sag_newline_pipe_1.nii.gz"),
        #         img_data, img)
        # # store roi_data
        # if roi.GetSpacing()[0] < 0.65 or roi.GetSpacing()[0] > 0.85:
        #     SaveNumpyToImageByRef(
        #         os.path.join(case_store_path, "womb_roi_forFeatureExtractor.nii"),
        #         roi_data, resampled_roi)
        # else:
        #     SaveNumpyToImageByRef(
        #         os.path.join(case_store_path, "cervix_roi.nii.gz"),
        #         roi_data, roi)

        # print(case_store_path)


if __name__ == "__main__":
    examine_and_store_data(
        root_path=r"E:\PhD\Data_renji\Data_forFeatureExtractor\41GE",
        store_path=r"E:\PhD\Data_renji\Data_forFeatureExtractor\41GE")
