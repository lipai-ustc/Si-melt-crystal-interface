import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# -------------------- 用户参数 --------------------
data_folder = r"E:\free_energy\MLSCAN\python_script\analysis\bond-length\111"
bins = 100               # 直方图分箱数
bond_min, bond_max = 1.8, 2.88
sigma = 2                # 高斯平滑强度
output_folder = os.path.join(data_folder, "combined_plot")
output_figure = os.path.join(output_folder, "combined_smoothed.png")

# 可以自定义每条线的名字，对应 npy 文件排序后的顺序
custom_labels = [
    "solid",
    "+3",
    "0",
    "-3",
    "-6",
    "liquid",
    # 根据实际文件数添加
]
# -------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# 找到所有 *_raw.npy 文件
npy_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
             if f.endswith("_raw.npy")]

if not npy_files:
    raise FileNotFoundError("没有找到 *_raw.npy 文件，请检查路径！")

plt.figure(figsize=(8, 5))
valid_files = 0

for idx, f in enumerate(sorted(npy_files)):
    try:
        bond_lengths = np.load(f)
        if bond_lengths.size == 0:
            print(f"跳过空文件: {f}")
            continue

        # ---- 归一化直方图 ----
        hist, bin_edges = np.histogram(bond_lengths, bins=bins,
                                       range=(bond_min, bond_max), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        integral = np.trapz(hist, bin_centers)
        print(f"{os.path.basename(f)}: 归一化检查 -> 积分 = {integral:.3f}")

        # ---- 高斯平滑 ----
        hist_smooth = gaussian_filter1d(hist, sigma=sigma)

        # ---- 自定义标签 ----
        if idx < len(custom_labels):
            label = custom_labels[idx]
        else:
            label = f"Slice {idx+1}"

        plt.plot(bin_centers, hist_smooth, label=label, lw=2)
        valid_files += 1

    except Exception as e:
        print(f"读取文件 {f} 出错: {e}")
        continue

if valid_files == 0:
    raise RuntimeError("没有任何有效数据可以绘制！")

# ---- 坐标轴设置 ----
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(1))  # 主刻度间隔，可根据实际调整
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.xlabel("Length (Å)", fontsize=20)
plt.ylabel("Probability", fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim(bond_min, bond_max)
plt.ylim(0, 3.3)  # 可根据需要调整
plt.legend(fontsize=20, loc="best", frameon=False)
plt.tight_layout()
plt.savefig(output_figure, dpi=300)
plt.show()

print(f"多条曲线已绘制并保存到: {output_figure}")
