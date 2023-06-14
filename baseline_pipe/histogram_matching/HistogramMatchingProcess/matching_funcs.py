""""
Several functions used in histogram matching
"""""
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import SimpleITK as sitk
from skimage.exposure import match_histograms


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(4096)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def slice2LUT(src_slice, ref_slice):
    """
    Get the lookup table(LUT) for each pair of src_cdf(src_slice), ref_cdf(ref_slice)
    :param src_slice:
    :param ref_slice:
    :return:
    """
    # rescale the slices
    rescaled_src_slice = (4095 * (src_slice - src_slice.min()) / (src_slice.max() - src_slice.min())).astype(np.uint16)
    rescaled_ref_slice = (4095 * (ref_slice - ref_slice.min()) / (ref_slice.max() - ref_slice.min())).astype(np.uint16)
    # histogram of the slices
    n_src, bins_src, patches_src = plt.hist(rescaled_src_slice.flatten(), bins=4096, range=[0, 4096], cumulative=True,
                                            density=True, rwidth=0.5, align="left")
    n_ref, bins_ref, patches_ref = plt.hist(rescaled_ref_slice.flatten(), bins=4096, range=[0, 4096], cumulative=True,
                                            density=True, rwidth=0.5, align="left")
    # change the dtype of n_src and n_ref
    lookup_table = calculate_lookup(n_src, n_ref).astype(np.uint16)
    matched_slice = match_histograms(rescaled_src_slice, rescaled_ref_slice).astype(np.uint16)

    return lookup_table, matched_slice


def average_mutipleLUTs(src_img_data, ref_img_data):
    """
    Average the multiple LUT curves given one case.
    :param src_img_data:
    :param ref_img_data:
    :return: look_up_table_arr-->mutipleLUTs[SlicesNum,4096], final_look_up_table-->averaged LUT
    """
    look_up_table_list = []
    slice_index_list = np.arange(len(src_img_data))
    for slice_index in slice_index_list:
        lookup_table, matched_slice = slice2LUT(src_img_data[slice_index], ref_img_data[slice_index])
        # plt.plot(lookup_table)  # , label=f"slice={slice_index}")
        # print("slice_index=",slice_index)
        # for index, item in enumerate(lookup_table):
        #     if item == 0:
        #         print(index)
        look_up_table_list.append(lookup_table)
    # further improvement：np.array([np.array(value) for value in look_up_table_list])
    look_up_table_arr = np.array(look_up_table_list, dtype=np.uint16)
    final_look_up_table = np.mean(look_up_table_arr, axis=0, dtype=np.uint16)
    # plt.plot(final_look_up_table, "ro")
    # plt.plot(np.arange(0, 4096), np.arange(0, 4096), "--", label="diagonal line")
    # plt.legend()
    # plt.title("The LUT for some slices", fontsize=14)
    # plt.xlim(left=0, right=4095)
    # plt.ylim(bottom=0, top=4095)
    # plt.xlabel("Intensity_source", fontsize=14)
    # plt.ylabel("Intensity_target", fontsize=14)
    # plt.show()
    return look_up_table_arr, final_look_up_table


def loop_all_cases(root_path):
    case_list = os.listdir(root_path)[:]
    for case in case_list:
        src_img_path = root_path + f"\\{case}\\t2sag_ori.nii.gz"
        ref_img_path = root_path + f"\\{case}\\t2sag_cycleGAN.nii.gz"
        src_img = sitk.ReadImage(src_img_path)
        ref_img = sitk.ReadImage(ref_img_path)
        src_img_arr = sitk.GetArrayFromImage(src_img)
        ref_img_arr = sitk.GetArrayFromImage(ref_img)
        look_up_table_arr, final_look_up_table = average_mutipleLUTs(src_img_arr, ref_img_arr)
        save_path = f".\\LUTresults\\{case}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(f"{save_path}\\LUTarr.npy", look_up_table_arr)
        np.save(f"{save_path}\\FinalLUT.npy", final_look_up_table)
        print(f"{case} Finished!")


if __name__ == "__main__":
    root_path = r"E:\PhD\Data_renji\Data_forFeatureExtractor\41GE"
    loop_all_cases(root_path)
