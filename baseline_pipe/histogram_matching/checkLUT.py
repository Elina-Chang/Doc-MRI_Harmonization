import matplotlib.pyplot as plt
import numpy as np
import cv2

from Tools.RootPath2CaseList import RootPath2CaseList
from Tools.Nii2Npy import Nii2Npy
from Tools.normalization1.normalization_after_clip import normalization_after_clip


def build_cdf(a, dcValues):
    """
    Function to get the cdf of an array
    Args:
        a (array): array to build the cdf from.
            If the shape is 1 dimensional, it is assumed to be a pdf
            If the shape is 2 dimensional, it is assumed to be a gray-scale img
        dcValues (int): maximum value of any element in the array
            For images this will be 255
    Returns:
        a single-dimension array cdf
    Raises:
        ValueError: image shape is not 2 (gray-scale) or 3 (color)
    """
    if (len(np.shape(a)) == 1):
        # it's a pdf
        pdf = a
    else:
        # get histogram from the image
        # first check if image is gray-scale or not
        if (len(np.shape(a)) == 2):
            # gray-scale image, look at channels [0]
            # args: images, channels, mask, histSize, ranges
            hist = cv2.calcHist([a], [0], None, [dcValues], [0, dcValues])
        else:
            raise ValueError("Invalid number of channels found: {}".format(
                len(np.shape(a))
            ))

        # get PDF of histogram
        # probability is the total histogram value, divided by total pixels
        pdf = hist / np.prod(np.shape(a))

    # get CDF of histogram
    cdf = np.cumsum(pdf)

    return cdf


def build_match_lookup_table(im, target, maxCount):
    """
    Function to build lookup table based on another image or pdf
    Args:
        im (array): image with initial histogram to be modified
        target (array): image or pdf to modify the im's histogram
        maxCount (int): maximum digital counts for the image
    Returns:
        a lookup table to perform on a histogram
    """

    # build the image cdf
    imgCDF = build_cdf(im, maxCount)

    # build the target (image or pdf) cdf
    tarCDF = build_cdf(target, maxCount)

    # build the lookup table
    # for every value, get the result from the imgCDF;
    # then, find the first index that exists for that value in the tarCDF.
    lut = np.arange(maxCount + 1)
    for i in range(maxCount):
        # sometimes our imgCDF is higher than the tarCDF ever is
        # lets put that at the maxCount
        if (imgCDF[i] > np.amax(tarCDF)):
            lut[i] = maxCount
        else:
            lut[i] = np.argmax(np.where(tarCDF >= imgCDF[i], 1, 0))

    return list(lut)


def plot_lut(lut):
    for (k, v) in lut.items():
        plt.plot(v)
        plt.title("LUT Curve")
    plt.legend(lut.keys())
    plt.show()


def lut_dict(src_path, tgt_path):
    src_array = Nii2Npy(src_path)
    clip_value, src_array = normalization_after_clip(src_array)  # Norm1
    src_array = 4095 * src_array

    tgt_array = Nii2Npy(tgt_path)
    tgt_array = 4095 * tgt_array

    lut = {}
    for slice_id in np.arange(len(src_array)):
        src_slice = src_array[slice_id]
        # print(src_slice.min(), src_slice.max())
        tgt_slice = tgt_array[slice_id]
        # print(tgt_slice.min(), tgt_slice.max())

        maxCount = int(np.amax(src_slice))
        lut["sliceID={}".format(slice_id)] = build_match_lookup_table(src_slice, tgt_slice, maxCount=maxCount)
    return lut


def main(root_path):
    case_list = RootPath2CaseList(root_path)
    random = np.random.randint(0, len(case_list))
    # print(random)
    src_path = case_list[random] + r"\t2_5x5.nii.gz"
    tgt_path = case_list[random] + r"\t2_5x5_matched.nii.gz"

    lut = lut_dict(src_path, tgt_path)
    plot_lut(lut)


if __name__ == "__main__":
    root_path = r"E:\PhD\Data_multivendor\UIH"
    main(root_path)
