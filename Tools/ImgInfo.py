import SimpleITK as sitk


def ImgInfo(data_directory):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()
    size = image3D.GetSize()
