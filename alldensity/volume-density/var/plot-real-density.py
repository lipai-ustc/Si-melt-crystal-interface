import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator

def plot_density_with_std_from_csv(data_path, output_path):
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 读取 sigma=0.8 的平均值 + 标准差
    sigma_std = 0.8
    file_std = os.path.join(data_path, f"density_avg_std_sigma{sigma_std}.csv")
    data_std = np.loadtxt(file_std, delimiter=',', skiprows=1)
    rel_grid = data_std[:, 0]
    mean_sigma_std = data_std[:, 1]
    std_sigma = data_std[:, 2]

    # 读取 sigma=0.08 的平均值
    sigma_low = 0.08
    file_low = os.path.join(data_path, f"density_avg_sigma{sigma_low}.csv")
    data_low = np.loadtxt(file_low, delimiter=',', skiprows=1)
    mean_sigma_low = data_low[:, 1]

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    fs = 23
    plot_x_min = -20
    plot_x_max = 20
    mask = (rel_grid >= plot_x_min) & (rel_grid <= plot_x_max)
    rel_grid_plot = rel_grid[mask]

    # sigma=0.8 阴影 ±σ
    ax.fill_between(rel_grid_plot,
                    mean_sigma_std[mask] - std_sigma[mask],
                    mean_sigma_std[mask] + std_sigma[mask],
                    color='C0', alpha=0.3, linewidth=0)
    # sigma=0.8 平均值曲线
    ax.plot(rel_grid_plot, mean_sigma_std[mask], color='C0', lw=2, alpha=0.9, label=f"σ={sigma_std} Å")

    # sigma=0.08 平均值曲线
    ax.plot(rel_grid_plot, mean_sigma_low[mask], color='C0', lw=2, alpha=0.2, label=f"σ={sigma_low} Å")

    # 红色虚线标界面
    #ax.axvline(0, color='r', linestyle='--')
    # ax.axvline(-10, color='r', linestyle='--')  # 可根据需求添加

    # 坐标轴标签与刻度
    ax.set_xlabel("z (Å)", fontsize=fs)
    ax.set_ylabel("Number density (Å⁻³)", fontsize=fs)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(0.049, 0.058)
    ax.yaxis.set_major_locator(MultipleLocator(0.002))
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

    # 其他绘图设置
    ax.grid(False)
    ax.legend(fontsize=fs - 4,frameon=False)

    # 保存图片
    output_file = os.path.join(output_path, f"density_sigma_0.8_0.08_plot.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    print(f"绘图完成，图片保存在 {output_file}")
    print("rel_grid_plot min:", rel_grid_plot.min())
    print("rel_grid_plot max:", rel_grid_plot.max())


if __name__ == "__main__":
    data_path = r"E:\free_energy\0all_new_pot\110\density\aim-vor-density"  # 数据文件目录
    output_path = r"E:\free_energy\0all_new_pot\110\density\aim-vor-density"  # 输出图片目录
    plot_density_with_std_from_csv(data_path, output_path)
