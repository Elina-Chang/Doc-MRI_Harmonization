from branches.new_normalization_on_GE.clip_and_normalization.clip.new_OtsuSegment import new_OtsuSegment


def generate_clip_value(data3D, bottom_clip_ratio=0.01, top_clip_ratio=0.005):
    mask = new_OtsuSegment(data3D)
    pixels = sorted(data3D[mask == 1].flatten())
    pixels = pixels[int(len(pixels) * bottom_clip_ratio):int(len(pixels) * (1 - top_clip_ratio))]

    bottom = pixels[0]
    top = pixels[-1]
    clip_value = {"bottom": bottom, "top": top}

    return clip_value


if __name__ == "__main__":
    pass
