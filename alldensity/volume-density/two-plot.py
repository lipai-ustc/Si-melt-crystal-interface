import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator


def plot_density_from_data(data_path, output_path):
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 定义要读取的sigma值
    sigmas = [0.8, 0.08]

    # 从文件读取数据
    data_dict = {}
    bins_dict = {}
    for sigma in sigmas:
        # 读取 density_relative_to_interface_sigma{sigma}.txt
        density_filename = f"density_relative_to_interface_sigma{sigma}.txt"
        density_path = os.path.join(data_path, density_filename)
        density_data = np.loadtxt(density_path)
        rel_grid = density_data[:, 0]
        density = density_data[:, 1]
        data_dict[sigma] = (rel_grid, density)

        # 读取 bins.txt
        bins_filename = f"biglq6_density_data.txt"
        bins_path = os.path.join(data_path, bins_filename)
        if os.path.exists(bins_path):
            bins_data = np.loadtxt(bins_path)
            rel_grid_bins = bins_data[:, 0]
            density_bins = bins_data[:, 1]
            bins_dict[sigma] = (rel_grid_bins, density_bins)
        else:
            print(f"警告：未找到 {bins_filename}，跳过该文件。")

    # 绘图设置
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    fs = 23  # 字体大小
    plot_x_min = -20
    plot_x_max = 20

    # 颜色和透明度
    line_color = 'C0'
    bins_color = 'C1'  # bins 用不同颜色
    alphas = {0.8: 0.9, 0.08: 0.2}

    # 绘制 density_relative_to_interface 数据
    for sigma in sigmas:
        if sigma not in data_dict:
            continue
        rel_grid, density = data_dict[sigma]
        mask = (rel_grid >= plot_x_min) & (rel_grid <= plot_x_max)
        ax.plot(
            rel_grid[mask],
            density[mask],
            color=line_color,
            alpha=alphas[sigma],
            label=f"σ={sigma} Å (density)"
        )

    # 绘制 bins.txt 数据
    for sigma in sigmas:
        if sigma not in bins_dict:
            continue
        rel_grid_bins, density_bins = bins_dict[sigma]
        mask = (rel_grid_bins >= plot_x_min) & (rel_grid_bins <= plot_x_max)
        ax.plot(
            rel_grid_bins[mask],
            density_bins[mask],
            color=bins_color,
            alpha=alphas[sigma],
            linestyle="--",  # bins 用虚线区分
            label=f"σ={sigma} Å (bins)"
        )

    # 设置坐标轴标签
    ax.set_xlabel("z (Å)", fontsize=fs)
    ax.set_ylabel("Number density (Å⁻³)", fontsize=fs)

    # 设置坐标轴范围和刻度
    #ax.set_ylim(0.049, 0.058)
   # y_major_locator = MultipleLocator(0.002)
    #ax.yaxis.set_major_locator(y_major_locator)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_xlim(plot_x_min, plot_x_max)

    ax.grid(False)
    ax.legend(fontsize=fs - 4)

    # 保存图片
    plt.tight_layout()
    output_file = os.path.join(output_path, "all_frames_sigma_0.5_0.2_density_with_bins.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    print(f"绘图完成，图片保存在 {output_file}")


if __name__ == "__main__":
    # 请修改为实际的数据路径和输出路径
    data_path = r"E:\free_energy\0all_new_pot\100\density\aim-vor-density"
    output_path = r"E:\free_energy\0all_new_pot\100\density\aim-vor-density\new"

    plot_density_from_data(data_path, output_path)
