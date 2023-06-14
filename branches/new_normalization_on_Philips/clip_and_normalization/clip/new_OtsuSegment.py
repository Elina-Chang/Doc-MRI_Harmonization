import cv2
from skimage import filters
import numpy as np


def new_OtsuSegment(img_data3D):
    img_data3D = (img_data3D - img_data3D.min()) / (img_data3D.max() - img_data3D.min())
    Th = filters.threshold_otsu(img_data3D)
    mask_array = np.where(img_data3D > Th / 6.1, 1, 0)
    mask_array = cv2.blur(mask_array, (12, 12))
    return mask_array
