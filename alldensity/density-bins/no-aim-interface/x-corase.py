import numpy as np
import matplotlib.pyplot as plt
import os
import re

######################################
# 用户输入参数：在这里改就好
######################################
# 只读取这些帧号（文件中 Frame=xxx）
start_frame =17
end_frame = 63
# 输入 xyz 文件路径
xyz_file = r"E:\free_energy\all_new_pot\111\cal-result\merge.xyz"

# 输出文件夹
output_dir = r"E:\free_energy\all_new_pot\111\density\fine-scale"

# 输出图片文件名
output_image_name_fine = f"fine_scale-{start_frame}-{end_frame}.png"
output_image_name_coarse = f"coarse_grained-{start_frame}-{end_frame}.png"
# 模拟盒子在 y 和 z 方向的尺寸 (Å)
#box_length_y = 53.7829
#box_length_z = 53.7829
#box_length_y = 52.8131
#box_length_z = 56.0168
box_length_y = 60.9397
box_length_z = 52.7753
# 沿 x 方向的分箱宽度 (Å)
bin_width = 0.05


# FIR 平滑参数
FIR_N = 200
FIR_epsilon = 75.0

# 如果想限制画图的 x 范围，就在这里设定
plot_x_min = 60  # 例如：10.0
plot_x_max = 100 # 例如：50.0
# 不想裁剪就保持 None

######################################
# 加载xyz中指定帧
######################################

def load_xyz_for_density(filename, start_frame, end_frame):
    frames = []
    current_frame_number = None

    with open(filename, 'r') as f:
        while True:
            # 第一行是原子数
            line = f.readline()
            if not line:
                break

            try:
                num_atoms = int(line.strip())
            except ValueError:
                print(f"[WARN] Unexpected line (not integer atom count): {line}")
                continue

            # 第二行包含 Frame=xxx
            comment_line = f.readline()
            frame_match = re.search(r'Frame\s*=\s*(\d+)', comment_line)
            if frame_match:
                current_frame_number = int(frame_match.group(1))
            else:
                print(f"[WARN] No Frame info in comment line: {comment_line.strip()}")
                current_frame_number = None

            # 检查是否在指定范围
            if current_frame_number is None or current_frame_number < start_frame or current_frame_number > end_frame:
                # 跳过
                for _ in range(num_atoms):
                    f.readline()
                continue

            # 读这个帧
            atoms = []
            for _ in range(num_atoms):
                parts = f.readline().strip().split()
                x = float(parts[1])
                atoms.append(x)

            frames.append(np.array(atoms))

    print(f"[INFO] Frames loaded in range [{start_frame}, {end_frame}]: {len(frames)}")
    return frames

######################################
# FIR 平滑
######################################

def fir_smooth(density_array, N=200, epsilon=75.0):
    k_vals = np.arange(-N, N+1)
    w = np.exp(-(k_vals/epsilon)**2)
    w /= np.sum(w)  # 归一化
    smoothed = np.convolve(density_array, w, mode='same')
    return smoothed

######################################
# 主绘图和保存
######################################

def compute_density(frames, bin_width, box_length_y, box_length_z):
    # 合并所有帧
    all_x = np.concatenate(frames)
    print(f"[INFO] Total x-coordinates from selected frames: {len(all_x)}")

    # 自动取边界
    x_min = np.min(all_x)
    x_max = np.max(all_x)
    bins = np.arange(x_min, x_max + bin_width, bin_width)

    # 统计
    counts, edges = np.histogram(all_x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # bin体积
    bin_volume = bin_width * box_length_y * box_length_z

    # 时间平均
    avg_counts_per_frame = counts / len(frames)

    # 变成number density
    number_density = avg_counts_per_frame / bin_volume

    return centers, number_density

def plot_and_save_density(centers, density, title, xlabel, ylabel, out_path,
                           x_min_plot=None, x_max_plot=None):
    plt.figure(figsize=(8,5))

    # 做裁剪
    if x_min_plot is not None or x_max_plot is not None:
        mask = np.ones_like(centers, dtype=bool)
        if x_min_plot is not None:
            mask &= (centers >= x_min_plot)
        if x_max_plot is not None:
            mask &= (centers <= x_max_plot)
        centers = centers[mask]
        density = density[mask]

    plt.plot(centers, density, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Plot saved to: {out_path}")

######################################
# 主流程
######################################

if __name__ == "__main__":
    print("[START] Loading xyz file...")
    frames = load_xyz_for_density(xyz_file, start_frame, end_frame)

    if not frames:
        print("[ERROR] No frames found in specified range!")
        exit()

    print("[START] Computing density profile...")
    centers, fine_density = compute_density(
        frames,
        bin_width=bin_width,
        box_length_y=box_length_y,
        box_length_z=box_length_z
    )

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------
    # 画 Fine-scale 版本
    # --------------------------
    plot_and_save_density(
        centers,
        fine_density,
        #title=f'Fine-scale density profile (Frames {start_frame}-{end_frame})',
        title=None,
        xlabel='x (Å)',
        ylabel='Number density (atoms/Å³)',
        out_path=os.path.join(output_dir, output_image_name_fine),
        x_min_plot=plot_x_min,
        x_max_plot=plot_x_max
    )

    # --------------------------
    # 生成和画 Coarse-grained 版本
    # --------------------------
    coarse_density = fir_smooth(fine_density, N=FIR_N, epsilon=FIR_epsilon)

    plot_and_save_density(
        centers,
        coarse_density,
        #title=f'Coarse-grained density profile (Frames {start_frame}-{end_frame})',
        title=None,
        xlabel='x (Å)',
        ylabel='Number density (atoms/Å³)',
        out_path=os.path.join(output_dir, output_image_name_coarse),
        x_min_plot=plot_x_min,
        x_max_plot=plot_x_max
    )

    print("[DONE] All finished!")
# 在交互窗口中画出 fine_density 结果
plt.figure(figsize=(8,5))
plt.plot(centers, fine_density, color='blue')
plt.xlabel('x (Å)')
plt.ylabel('Number density (atoms/Å³)')
plt.title(f'Fine-scale density profile (Frames {start_frame}-{end_frame})')

# 可选：限制范围
if plot_x_min is not None or plot_x_max is not None:
    plt.xlim(plot_x_min, plot_x_max)

# 自定义坐标提示格式
ax = plt.gca()
def format_coord(x, y):
    return f"x={x:.3f} Å, density={y:.5f} atoms/Å³"
ax.format_coord = format_coord

plt.tight_layout()
plt.show()
