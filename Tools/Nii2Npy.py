import SimpleITK as sitk


def Nii2Npy(path):
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    return img, array
