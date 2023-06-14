import numpy as np
import seaborn as sns
from scipy.ndimage.morphology import binary_dilation

color_list = sns.color_palette('bright') + sns.color_palette('deep')


def MergeImageWithROI(data, roi, overlap=False):
    global index_x, index_y
    if data.max() > 1.0:
        print('Scale the data manually.')
        return data

    if data.ndim >= 3:
        print("Should input 2d image")
        return data

    if not isinstance(roi, list):
        roi = [roi]

    if len(roi) > len(color_list):
        print('There are too many ROIs')
        return data

    intensity = 255
    data = np.asarray(data * intensity, dtype=np.uint8)

    if overlap:
        new_data = np.stack([data, data, data], axis=2)
        for one_roi, color in zip(roi, color_list[:len(roi)]):
            index_x, index_y = np.where(one_roi == 1)
            new_data[index_x, index_y, :] = np.asarray(color) * intensity
    else:
        kernel = np.ones((3, 3))
        new_data = np.stack([data, data, data], axis=2)
        for one_roi, color in zip(roi, color_list[:len(roi)]):
            boundary = binary_dilation(input=one_roi, structure=kernel, iterations=1) - one_roi
            index_x, index_y = np.where(boundary == 1)
            new_data[index_x, index_y, :] = np.asarray(color) * intensity
    return new_data, index_x, index_y
