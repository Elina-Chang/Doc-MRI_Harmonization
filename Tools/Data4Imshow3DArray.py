import numpy as np
from Tools.MinMaxScale import MinMaxScale


def Data4Imshow3DArray(array, isFlipud=False, isFliplr=False):
    array = np.transpose(array, [1, 2, 0])
    array = MinMaxScale(array)
    if isFlipud:
        array = np.flipud(array)
    if isFliplr:
        array = np.fliplr(array)
    return array
