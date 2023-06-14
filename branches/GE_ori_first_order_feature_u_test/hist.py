import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

plt.hist([1, 3, 3, 2, 0, 0, 1, 3, 0, 5, 4, 0, 0, 0, 1, 1, 6, 1, 0, 0], bins=20, align="mid")
plt.title("GE组内一阶特征差异性统计(3T+womb)", fontsize=16)
plt.xlabel("具有显著性差异的特征数", fontsize=14)
plt.ylabel("出现次数", fontsize=14)
plt.xticks(np.arange(0, 7, step=1))
plt.yticks(np.arange(0, 9, step=1))
plt.savefig("womb_result.png")
plt.show()
