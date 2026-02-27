import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# -------------------- 用户参数 --------------------
data_folder = r"E:\free_energy\MLSCAN\python_script\analysis\0slice-analysis\entropy\110"  # realentropy 的 npy 文件所在文件夹
bins = 100  # 直方图分箱数

# ⚠️ 根据实际 realentropy 范围修改
realentropy_min, realentropy_max = -2, -0.6  # 例如根据你发的文件可设置这个范围

sigma = 2  # 高斯平滑强度
output_folder = os.path.join(data_folder, "combined_plot")
output_figure = os.path.join(output_folder, "combined_realentropy.png")

# 自定义每条曲线的名字
custom_labels = [
    "solid",
    "slice0_+3",
    "slice-3_0",
    "slice-6_-3",
    "slice-9_-6",
    "liquid"
]
# -------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# 找到所有 *_x*_*.npy 文件（realentropy）
npy_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
             if f.endswith(".npy") and "realentropy" in f]

if not npy_files:
    raise FileNotFoundError("没有找到 realentropy npy 文件，请检查路径！")

plt.figure(figsize=(8, 5))
valid_files = 0

for idx, f in enumerate(sorted(npy_files)):
    try:
        realentropy_data = np.load(f)
        if realentropy_data.size == 0:
            print(f"跳过空文件: {f}")
            continue

        # ---- 归一化直方图 ----
        hist, bin_edges = np.histogram(realentropy_data, bins=bins,
                                       range=(realentropy_min, realentropy_max), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        integral = np.trapz(hist, bin_centers)
        print(f"{os.path.basename(f)}: 归一化检查 -> 积分 = {integral:.3f}")

        # ---- 高斯平滑 ----
        hist_smooth = gaussian_filter1d(hist, sigma=sigma)

        # ---- 标签 ----
        label = custom_labels[idx] if idx < len(custom_labels) else f"Slice {idx+1}"
        plt.plot(bin_centers, hist_smooth, label=label, lw=2)
        valid_files += 1

    except Exception as e:
        print(f"读取文件 {f} 出错: {e}")
        continue

if valid_files == 0:
    raise RuntimeError("没有任何有效数据可以绘制！")

# ---- 坐标轴设置 ----
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.xlabel("Real Entropy", fontsize=17)
plt.ylabel("Probability", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(realentropy_min, realentropy_max)
plt.ylim(0, 4)  # 可根据实际数据调整
plt.legend(fontsize=13, loc="best", frameon=False)
plt.tight_layout()
plt.savefig(output_figure, dpi=300)
plt.show()

print(f"多条归一化 Real Entropy 曲线已绘制并保存到: {output_figure}")
