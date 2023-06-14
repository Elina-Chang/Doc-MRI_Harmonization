import SimpleITK as sitk
import os
import numpy as np
from copy import deepcopy

class Resampler():
    def __init__(self):
        pass

    def _GenerateFileName(self, file_path, name):
        store_path = ''
        if os.path.splitext(file_path)[1] == '.nii':
            store_path = file_path[:-4] + '_' + name + '.nii'
        elif os.path.splitext(file_path)[1] == '.gz':
            store_path = file_path[:-7] + '_' + name + '.nii.gz'
        else:
            print('the input file should be suffix .nii or .nii.gz')

        return store_path

    def ResizeSipmleITKImage(self, image, is_roi=False, expected_resolution=[], expected_shape=[], method=sitk.sitkBSpline,
                             dtype=sitk.sitkFloat32, store_path=''):
        if (expected_resolution == []) and (expected_shape == []):
            print('Give at least one parameters. ')
            return image

        if isinstance(image, str) and os.path.exists(image):
            image_path = deepcopy(image)
            image = sitk.ReadImage(image)
        else:
            image_path = ''

        shape = image.GetSize()
        resolution = image.GetSpacing()

        if expected_resolution == []:
            dim_0, dim_1, dim_2 = False, False, False
            if expected_shape[0] == 0:
                expected_shape[0] = shape[0]
                dim_0 = True
            if expected_shape[1] == 0:
                expected_shape[1] = shape[1]
                dim_1 = True
            if expected_shape[2] == 0:
                expected_shape[2] = shape[2]
                dim_2 = True
            expected_resolution = [raw_resolution * raw_size / dest_size for dest_size, raw_size, raw_resolution in
                                   zip(expected_shape, shape, resolution)]
            if dim_0: expected_resolution[0] = resolution[0]
            if dim_1: expected_resolution[1] = resolution[1]
            if dim_2: expected_resolution[2] = resolution[2]

        elif expected_shape == []:
            dim_0, dim_1, dim_2 = False, False, False
            if expected_resolution[0] < 1e-6:
                expected_resolution[0] = resolution[0]
                dim_0 = True
            if expected_resolution[1] < 1e-6:
                expected_resolution[1] = resolution[1]
                dim_1 = True
            if expected_resolution[2] < 1e-6:
                expected_resolution[2] = resolution[2]
                dim_2 = True
            expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                              dest_resolution, raw_size, raw_resolution in zip(expected_resolution, shape, resolution)]
            if dim_0: expected_shape[0] = shape[0]
            if dim_1: expected_shape[1] = shape[1]
            if dim_2: expected_shape[2] = shape[2]

        resample_filter = sitk.ResampleImageFilter()

        if is_roi:
            temp_output = resample_filter.Execute(image, expected_shape, sitk.AffineTransform(len(shape)), sitk.sitkLinear,
                                             image.GetOrigin(), expected_resolution, image.GetDirection(), 0.0, dtype)
            roi_data = sitk.GetArrayFromImage(temp_output)

            new_data = np.zeros(roi_data.shape, dtype=np.uint8)
            pixels = np.unique(sitk.GetArrayFromImage(image))
            for i in range(len(pixels)):
                if i == (len(pixels) - 1):
                    max = pixels[i]
                    min = (pixels[i - 1] + pixels[i]) / 2
                elif i == 0:
                    max = (pixels[i] + pixels[i + 1]) / 2
                    min = pixels[i]
                else:
                    max = (pixels[i] + pixels[i + 1]) / 2
                    min = (pixels[i - 1] + pixels[i]) / 2
                new_data[np.bitwise_and(roi_data > min, roi_data <= max)] = pixels[i]
            output = sitk.GetImageFromArray(new_data)
            output.CopyInformation(temp_output)
        else:
            output = resample_filter.Execute(image, expected_shape, sitk.AffineTransform(len(shape)), method,
                                             image.GetOrigin(), expected_resolution, image.GetDirection(), 0.0, dtype)

        if store_path and image_path:
            sitk.WriteImage(output, self._GenerateFileName(image_path, 'Resample'))
        elif store_path and store_path.endswith(('.nii', '.nii.gz')):
            sitk.WriteImage(output, store_path)

        return output
