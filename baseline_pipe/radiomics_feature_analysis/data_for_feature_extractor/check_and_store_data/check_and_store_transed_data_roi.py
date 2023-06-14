import SimpleITK as sitk
import numpy as np
import glob
import os
from cyclegan_mri.Radiomics.Visualization import Imshow3DArray
from cyclegan_mri.Radiomics.SaveModel import SaveNumpyToImageByRef
from cyclegan_mri.Resampler import Resampler

resampler = Resampler()


def examine_and_store_data(img_root_path, roi_root_path, store_path):
    cases_path_img = glob.glob(img_root_path + "\*")
    cases_path_roi = glob.glob(roi_root_path + "\*")
    for case_img, case_roi in zip(cases_path_img, cases_path_roi):
        print(os.path.basename(case_img))

        img_path = glob.glob(case_img + r"\*t2sag.nii.gz")[0]
        img = sitk.ReadImage(img_path)
        if img.GetSpacing()[0] < 0.75 or img.GetSpacing()[0] > 0.85:
            resampled_img = resampler.ResizeSipmleITKImage(image=img_path, is_roi=False,
                                                           expected_resolution=[0.75, 0.75, -1])
            img_data = sitk.GetArrayFromImage(resampled_img)
        else:
            img_data = sitk.GetArrayFromImage(img)
        img_data = np.transpose(img_data, [1, 2, 0])
        print(img_data.shape, img_data.dtype, img_data.min(), img_data.max())
        # img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min()) #结果证明这种归一化方式不可取。
        # 换归一化方式。
        # from baseline_pipe.model_prediction.predict_on_output02.normalization1.normalization_after_clip import normalization_after_clip
        # clip_value, img_data = normalization_after_clip(img_data)
        # print(img_data.shape, img_data.dtype, img_data.min(), img_data.max())
        # img_data = np.flipud(img_data)

        roi_path = glob.glob(case_roi + r"\*sag_Merge.nii")[0]
        roi = sitk.ReadImage(roi_path)
        if roi.GetSpacing()[0] < 0.75 or roi.GetSpacing()[0] > 0.85:  # resample roi_data
            resampled_roi = resampler.ResizeSipmleITKImage(image=roi_path, is_roi=True,
                                                           expected_resolution=[0.75, 0.75, -1])
            roi_data = sitk.GetArrayFromImage(resampled_roi)
        else:
            roi_data = sitk.GetArrayFromImage(roi)
        roi_data = roi_data.astype(dtype=np.uint8)
        roi_data = np.transpose(roi_data, [1, 2, 0])
        print(roi_data.shape, roi_data.dtype, roi_data.min(), roi_data.max())
        roi_data = np.flipud(roi_data)

        # Imshow3DArray(data=img_data, roi=roi_data)

        # store img_data and roi_data
        case_store_path = os.path.join(store_path, os.path.basename(case_img))
        if not os.path.exists(case_store_path):
            os.makedirs(case_store_path)
        # store img_data
        if img.GetSpacing()[0] < 0.75 or img.GetSpacing()[0] > 0.85:
            SaveNumpyToImageByRef(
                os.path.join(case_store_path, "t2sag_ori.nii.gz"),
                img_data, resampled_img)
        else:
            SaveNumpyToImageByRef(
                os.path.join(case_store_path, "t2sag_ori.nii.gz"),
                img_data, img)
        # # store roi_data
        # if roi.GetSpacing()[0] < 0.75 or roi.GetSpacing()[0] > 0.85:
        #     SaveNumpyToImageByRef(
        #         os.path.join(case_store_path, "cervix_roi.nii.gz"),
        #         roi_data, resampled_roi)
        # else:
        #     SaveNumpyToImageByRef(
        #         os.path.join(case_store_path, "cervix_roi.nii.gz"),
        #         roi_data, roi)

        # print(case_store_path)


if __name__ == "__main__":
    examine_and_store_data(img_root_path=r"G:\PhD\Data_renji\Data_3T_Resampled_Norm1\UIH_3T_Resampled_Norm1",
                           roi_root_path=r"C:\Users\Administrator\Desktop\UIH",
                           store_path=r"G:\PhD\Data_renji\Data_forFeatureExtractor_\7UIH")
