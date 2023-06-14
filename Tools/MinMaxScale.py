def MinMaxScale(array):
    array = (array - array.min()) / (array.max() - array.min())
    return array
