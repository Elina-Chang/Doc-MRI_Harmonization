from Tools.normalization1.generate_clip_value import generate_clip_value


def normalization_after_clip(data3D):
    clip_value = generate_clip_value(data3D)
    data3D[data3D < clip_value["bottom"]] = clip_value["bottom"]
    data3D[data3D > clip_value["top"]] = clip_value["top"]
    data3D = (data3D - data3D.min()) / (data3D.max() - data3D.min())
    return clip_value, data3D
