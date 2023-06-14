import os
import numpy as np
import matplotlib.pyplot as plt

from Tools.Merge2DImageWithROI import MergeImageWithROI


def ret_mid_slice(case_path="", nii_name=""):
    from Tools.Nii2Npy import Nii2Npy
    img, img_arr = Nii2Npy(os.path.join(case_path, nii_name))
    return img_arr[len(img_arr) // 2]


def ret_slice(case_path="", nii_name="", slice_no=0):
    from Tools.Nii2Npy import Nii2Npy
    img, img_arr = Nii2Npy(os.path.join(case_path, nii_name))
    return img_arr[slice_no]


def save_slice(slice, enlarge=False, index_x=None, index_y=None, save_path="", title=""):
    if enlarge:
        x_range = [np.min(index_x) - 15, np.max(index_x) + 15]
        y_range = [np.min(index_y) - 15, np.max(index_y) + 15]
        slice = slice[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    plt.imshow(slice, cmap="gray")
    plt.axis("off")
    # plt.show()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{title}.svg"), dpi=600, bbox_inches="tight", pad_inches=0.0)


def psnr_mssim_nmae(y_true, y_predict):
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    from sklearn.metrics import mean_absolute_error
    psnr = peak_signal_noise_ratio(y_true, y_predict)
    mssim = structural_similarity(y_true, y_predict)
    nmae = mean_absolute_error(y_true, y_predict) / np.mean(y_true)
    return psnr, mssim, nmae


if __name__ == "__main__":
    # GE
    ori_case_path = r"E:\PhD\Data_renji\Data_3T_Resampled_Norm1\GE_3T_Resampled_Norm1\TANG LIU RONG_0524214"
    trans_case_path = r"E:\PhD\StyleTransferPredictions\modified_05_tvloss\41GEpredictions\TANG LIU RONG_0524214"
    img_slice_ori = ret_mid_slice(case_path=ori_case_path, nii_name="t2sag.nii.gz")
    img_slice_trans = ret_mid_slice(case_path=trans_case_path, nii_name="t2sag_PhilipsStyle.nii.gz")
    save_slice(slice=img_slice_ori, save_path=r".\GEcase_TANG LIU RONG_0524214", title="Input")
    # psnr, mssim, nmae = psnr_mssim_nmae(y_true=img_slice_ori, y_predict=img_slice_trans)
    # print(f"psnr:{psnr},mssim:{mssim},nmae:{nmae}")

    # # GE
    # trans_case_path = r"E:\PhD\Data_renji\Data_forFeatureExtractor\41GE\REN SU QIN_2864141"
    # img_slice_trans = ret_slice(case_path=trans_case_path, nii_name="t2sag_ori.nii", slice_no=12)
    # roi_slice = ret_slice(case_path=ori_case_path, nii_name="cervix_roi.nii.gz", slice_no=12)
    # img_roi_slice, index_x, index_y = MergeImageWithROI(data=img_slice_trans, roi=roi_slice)
    # save_slice(img_roi_slice, enlarge=False, index_x=index_x, index_y=index_y, save_path=r".\REN SU QIN_2864141",
    #            title="Input")

    # # Philips
    # ori_case_path = r"E:\PhD\Data_renji\Data_forFeatureExtractor\89Philips\REN_WEI_2760261"
    # img_slice_ori = ret_slice(case_path=ori_case_path, nii_name="t2sag_ori.nii", slice_no=16)
    # roi_slice = ret_slice(case_path=ori_case_path, nii_name="cervix_roi.nii.gz", slice_no=16)
    # img_roi_slice, index_x, index_y = MergeImageWithROI(data=img_slice_ori, roi=roi_slice)
    #
    # save_slice(img_roi_slice, enlarge=True, index_x=index_x, index_y=index_y, save_path=r".\REN_WEI_2760261",
    #            title="Input(ROI)")
