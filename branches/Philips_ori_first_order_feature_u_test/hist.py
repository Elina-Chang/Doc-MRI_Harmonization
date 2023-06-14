import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

plt.hist([[3, 0, 1, 2, 4, 0, 0, 3, 6, 0, 0, 0, 2, 0, 0, 9, 1, 0, 0, 1]], bins=20, align="mid")
plt.title("Philips组内一阶特征差异性统计", fontsize=16)
plt.xlabel("具有显著性差异的特征数", fontsize=14)
plt.ylabel("出现次数", fontsize=14)
plt.xticks(np.arange(0, 10, step=1))
plt.yticks(np.arange(0, 11, step=1))
plt.savefig("womb_result.png")
plt.show()
