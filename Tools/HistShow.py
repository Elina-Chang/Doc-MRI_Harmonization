import matplotlib.pyplot as plt


def HistShow(pixels_list=[]):
    plt.hist(pixels_list, bins=100)
    plt.show()
