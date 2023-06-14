import cv2
from skimage import filters
import SimpleITK as sitk
import numpy as np

from branches.new_normalization_on_GE.clip_and_normalization.Visualization import Imshow3DArray


def get_img_data(img_path):
    img = sitk.ReadImage(img_path)
    img_data = sitk.GetArrayFromImage(img)
    img_data = np.transpose(img_data, (1, 2, 0))
    img_data = np.flipud(img_data)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    return img_data


def new_OtsuSegment(img_data3D):
    Th = filters.threshold_otsu(img_data3D)
    mask_array = np.where(img_data3D > Th / 6.1, 1, 0)
    mask_array = cv2.blur(mask_array, (12, 12))
    return mask_array


def get_random_img_path(root_path=r"G:\PhD\Data_renji\GE_3T_resampled"):
    import os
    cases_list = os.listdir(root_path)
    path_list = [os.path.join(root_path, case_name, "t2sag_resampled.nii") for case_name in cases_list]
    import random
    index = random.randint(0, 40)
    print(index)
    img_path = path_list[index]
    return img_path


def plot_slice(slice):
    import matplotlib.pyplot as plt
    plt.imshow(slice, cmap="gray")
    plt.show()


if __name__ == "__main__":
    img_path = get_random_img_path()
    img_data = get_img_data(img_path)
    mask_array = new_OtsuSegment(img_data)
    # print(img_data.shape, mask_array.shape)
    # plot_slice(mask_array[...,0])
    Imshow3DArray(img_data, mask_array)
