import random

from Tools.Nii2Npy import Nii2Npy
from Tools.Data4Imshow3DArray import Data4Imshow3DArray
from Tools.Visualization import Imshow3DArray


def RandShowImgRoi(cases_list, img_name, roi_name, isFlipud=False, isFliplr=False):
    random_no = random.randint(0, len(cases_list))
    for case in cases_list[random_no:]:
        print("case's name: %s" % case)
        img, img_array = Nii2Npy(case + r"\{}".format(img_name))
        roi, roi_array = Nii2Npy(case + r"\{}".format(roi_name))
        img_array = Data4Imshow3DArray(img_array, isFlipud=isFlipud, isFliplr=isFliplr)
        roi_array = Data4Imshow3DArray(roi_array, isFlipud=isFlipud, isFliplr=isFliplr)
        Imshow3DArray(img_array, roi_array)


if __name__ == "__main__":
    # test
    root_path = r"E:\PhD\Data_renji\Data_forFeatureExtractor\GE"
    from Tools.RootPath2CaseList import RootPath2CaseList

    case_list = RootPath2CaseList(root_path)
    RandShowImgRoi(case_list, "t2sag_clahed.nii.gz", "cervix_roi.nii.gz")
