import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from cyclegan_mri.Radiomics.Visualization import Imshow3DArray


def ShowDataFromNii(path):
    # Show data from .nii format file
    image = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(image)
    N = len(data)
    fig, axes = plt.subplots(nrows=6, ncols=N // 6 + 1, figsize=(10, 10))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for i, image in enumerate(data):
        axes[i].imshow(image, cmap="gray")
    plt.show()


def ShowDataFromNpy(data):
    # Show data from numpy array
    N = data.shape[0]
    fig, axes = plt.subplots(nrows=6, ncols=N // 6 + 1, figsize=(10, 10))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for i, image in enumerate(data):
        axes[i].imshow(image, cmap="gray")
    plt.show()


def show_data_and_roi(data_path, roi_path):
    # Show data and its roi from .npy format file
    data = np.load(data_path)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))  # 归一化
    roi = np.load(roi_path)
    print(data.shape, data.dtype)
    Imshow3DArray(data, roi)


def show_data_and_roi2(data_path, roi_path):
    # Show data and its roi from .nii format file
    data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
    data = np.transpose(data, [1, 2, 0])
    data = (data - np.min(data)) / (np.max(data) - np.min(data))  # 归一化
    roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))
    roi = np.transpose(roi, [1, 2, 0])
    Imshow3DArray(data, roi)


if __name__ == "__main__":
    data_path = r"E:\PhD\PycharmProjects\my_project\CycleGAN_MRI\CycleGAN_MRI\all_data\GE_womb_t2sag" \
                r"\ZHUO_QIU_LAN_2440697\matched_t2_sag_with_train3'_partial_data_log01.nii"
    show_data(data_path)
