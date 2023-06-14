from __future__ import print_function
import numpy as np
from scipy.ndimage.morphology import binary_dilation
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import seaborn as sns

color_list = sns.color_palette('bright') + sns.color_palette('deep')

def MergeImageWithROI(data, roi, overlap=False):
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
    return new_data

def Merge3DImageWithROI(data, roi, overlap=False):
    if not isinstance(roi, list):
        roi = [roi]

    # if len(roi) > 3:
    #     print('Only show 3 ROIs')
    #     return data

    new_data = np.zeros((data.shape[2], data.shape[0], data.shape[1], 3))
    for slice_index in range(data.shape[2]):
        slice = data[..., slice_index]
        one_roi_list = []
        for one_roi in roi:
            one_roi_list.append(one_roi[..., slice_index])

        new_data[slice_index, ...] = MergeImageWithROI(slice, one_roi_list, overlap=overlap)

    return new_data

def Imshow3DArray(data, roi=None, window_size=[800, 800], window_name='Imshow3D', overlap=False):
    '''
    Imshow 3D Array, the dimension is row x col x slice. If the ROI was combined in the data, the dimension is:
    slice x row x col x color
    :param data: 3D Array [row x col x slice] or 4D array [slice x row x col x RGB]
    '''
    if isinstance(roi, list) or isinstance(roi, type(data)):
        data = Merge3DImageWithROI(data, roi, overlap=overlap)

    if np.ndim(data) == 3:
        data = np.swapaxes(data, 0, 1)
        data = np.transpose(data)

    pg.setConfigOptions(imageAxisOrder='row-major')
    app = QtGui.QApplication([])

    win = QtGui.QMainWindow()
    win.resize(window_size[0], window_size[1])
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle(window_name)

    imv.setImage(data)
    app.exec()