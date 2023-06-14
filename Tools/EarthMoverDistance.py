import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats

"""
    # Calculate Earth-Mover Distance
"""


# Get the frequencies in histogram and save
def get_relative_frequencies(pixels, need_save_relative_frequencies=False, save_name=""):
    n, bins, patches = plt.hist(pixels, bins=10)
    frequencies = n
    relative_frequencies = frequencies / frequencies.sum()  # frequency ——> relative frequency
    if need_save_relative_frequencies:
        np.save(f"{save_name}.npy", relative_frequencies)
    return relative_frequencies  # 返回直方图的纵坐标（频率表示）


# Earth mover distance
def earth_mover_distance(distribution_x, distribution_y):
    distance = stats.wasserstein_distance(distribution_x, distribution_y)
    return distance
