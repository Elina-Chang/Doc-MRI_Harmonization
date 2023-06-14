import cv2
from skimage import filters
import numpy as np


def new_OtsuSegment(img_array):
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    Th = filters.threshold_otsu(img_array)
    mask_array = np.where(img_array > Th / 6.1, 1, 0)
    mask_array = cv2.blur(mask_array, (12, 12))
    return mask_array


if __name__ == "__main__":
    import SimpleITK as sitk
    from Tools.Data4Imshow3DArray import Data4Imshow3DArray
    from Tools.Visualization import Imshow3DArray

    data = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\Elina\Desktop\t2sag_ori.nii"))
    mask_array = new_OtsuSegment(data)
    img_array = Data4Imshow3DArray(data, isFlipud=False)
    roi_array = Data4Imshow3DArray(mask_array, isFlipud=True)
    Imshow3DArray(img_array)
