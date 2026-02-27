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
    for sigma in sigmas:
        filename = f"density_relative_to_interface_sigma{sigma}.txt"
        file_path = os.path.join(data_path, filename)
        data = np.loadtxt(file_path)
        # 提取x和密度数据
        rel_grid = data[:, 0]
        density = data[:, 1]
        data_dict[sigma] = (rel_grid, density)

    # 绘图设置，与原代码保持一致
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    fs = 23  # 字体大小
    plot_x_min = -20
    plot_x_max = 20

    # 统一颜色，不同透明度
    line_color = 'C0'
    alphas = {0.8: 1, 0.08: 0.4}

    # 绘制每条曲线
    for sigma in sigmas:
        rel_grid, density = data_dict[sigma]
        # 应用x范围掩码
        mask = (rel_grid >= plot_x_min) & (rel_grid <= plot_x_max)
        rel_grid_plot = rel_grid[mask]
        density_plot = density[mask]

        ax.plot(
            rel_grid_plot,
            density_plot,
            color=line_color,
            alpha=alphas[sigma],
            label=f"σ={sigma} Å"
        )

    # 绘制红色虚线
    #ax.axvline( -7.16792, color='r', linestyle='--')
    #ax.axvline(-1.15288, color='r', linestyle='--')

    # 设置坐标轴标签
    ax.set_xlabel("z (Å)", fontsize=fs)
    ax.set_ylabel("Number density (Å⁻³)", fontsize=fs)

    # 设置坐标轴范围和刻度
    ax.set_ylim(0.049, 0.058)
    y_major_locator = MultipleLocator(0.002)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    ax.set_xlim(plot_x_min, plot_x_max)

    # 其他绘图设置
    ax.grid(False)
    ax.legend(fontsize=fs - 4, frameon=False)


    # 保存图片
    plt.tight_layout()
    output_file = os.path.join(output_path, "all_frames_sigma_0.5_0.2_density.png")
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.01
    )
    plt.close()

    # 打印一些信息
    rel_grid_plot = data_dict[sigmas[0]][0][mask]
    print("rel_grid_plot min:", rel_grid_plot.min())
    print("rel_grid_plot max:", rel_grid_plot.max())
    print(f"绘图完成，图片保存在 {output_file}")


if __name__ == "__main__":
    # 请修改为实际的数据路径和输出路径
    data_path = r"E:\free_energy\0all_new_pot\111\density\aim-vor-density"  # 存放数据文件的目录
    output_path = r"E:\free_energy\0all_new_pot\111\density\aim-vor-density"  # 图片输出目录

    plot_density_from_data(data_path, output_path)
