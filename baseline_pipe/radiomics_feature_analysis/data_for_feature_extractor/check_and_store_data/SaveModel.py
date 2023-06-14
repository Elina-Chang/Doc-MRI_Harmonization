from pathlib import Path
import SimpleITK as sitk
import numpy as np


def GetImageFromArrayByImage(data, refer_image, is_transfer_axis=True, flip_log=[0, 0, 0]):
    if is_transfer_axis:
        data = np.swapaxes(data, 0, 1)
        for index, is_flip in enumerate(flip_log):
            if is_flip:
                data = np.flip(data, axis=index)
        data = np.transpose(data)
    new_image=sitk.GetImageFromArray(data)
    new_image.CopyInformation(refer_image)
    return new_image


def SaveNumpyToImageByRef(store_path, data, ref_image):
    if isinstance(store_path, Path):
        store_path = str(store_path)
    image = GetImageFromArrayByImage(data, ref_image)
    image.CopyInformation(ref_image)
    sitk.WriteImage(image, store_path)
