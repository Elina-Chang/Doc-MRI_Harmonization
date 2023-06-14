import cv2
import numpy as np
import SimpleITK as sitk

from Tools.RootPath2CaseList import RootPath2CaseList
from Tools.Nii2Npy import Nii2Npy
from Tools.normalization1.normalization_after_clip import normalization_after_clip


def get_data_path(root_dir=""):
    case_list = RootPath2CaseList(root_dir)
    data_path_list = [case + r"\t2sag_ori.nii" for case in case_list]
    return data_path_list


def equalize(slice, mode="clahe"):
    """
    :param slice:
    :param mode: be "plain" or "clahe".
    :return:
    """
    if mode == "plain":
        eq_slice = cv2.equalizeHist(slice)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        eq_slice = clahe.apply(slice)
    return eq_slice


def show_func(img1, img2):
    import matplotlib.pyplot as plt
    from Tools.normalization1.new_OtsuSegment import new_OtsuSegment
    mask1 = new_OtsuSegment(img1)
    mask2 = new_OtsuSegment(img2)

    plt.subplot(221)
    plt.imshow(img1, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(222)
    plt.imshow(img2, cmap="gray")
    plt.title("Image after CLAHE")
    plt.axis("off")

    plt.subplot(212)
    plt.hist(img1[mask1 == 1].flat, bins=100, density=True, histtype="step", label='Original Histogram')
    plt.hist(img2[mask2 == 1].flat, bins=100, density=True, histtype="step", label='Histogram after CLAHE')
    plt.legend(loc='upper right')
    plt.xlim(left=0, right=260)
    plt.xticks(np.arange(0, 300, 50))

    # plt.subplot(224)
    # plt.hist(img2[mask2 == 1].flat, bins=100, density=True)
    # plt.xlim(left=0, right=260)
    # plt.xticks(np.arange(0, 300, 50))
    plt.show()


def main(root_dir="", store_root_dir=""):
    data_path_list = get_data_path(root_dir)
    # import random
    # random.shuffle(data_path_list)
    for data_path in data_path_list:
        img, array = Nii2Npy(data_path)
        output_list = []
        for slice_index, slice in enumerate(array):
            slice = (255 * slice).astype(np.uint8)
            eq_slice = equalize(slice, mode="clahe")
            # show_func(slice, eq_slice)
            output_list.append(eq_slice)
        output4store = np.stack(output_list, axis=0)
        output4store = ((output4store - output4store.min()) / (output4store.max() - output4store.min())).astype(
            np.float32)  # Min-max Norm
        # _, output4store = normalization_after_clip(output4store)  # Norm1
        store_img = sitk.GetImageFromArray(output4store)
        store_img.CopyInformation(img)
        case_name = data_path.split("\\")[-2]
        store_path = store_root_dir + "\\" + case_name + "\\t2sag_clahed.nii.gz"
        sitk.WriteImage(store_img, store_path)


if __name__ == "__main__":
    data_path_list = main(r"E:\PhD\Data_renji\Data_forFeatureExtractor\PHILIPS",
                          r"E:\PhD\Data_renji\Data_forFeatureExtractor\PHILIPS")
