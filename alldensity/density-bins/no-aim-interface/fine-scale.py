import numpy as np
import matplotlib.pyplot as plt
import os

######################################
# 用户输入参数：在这里改就好
######################################

# 输入 xyz 文件路径
xyz_file = r"E:\free_energy\all_new_pot\100\cal-result\merge.xyz"

# 输出文件夹
output_dir = r"E:\free_energy\all_new_pot\100\density\fine-scale"

# 输出图片名字
output_image_name = "fine_scale_density_x.png"

# 模拟盒子在 y 和 z 方向的尺寸 (Å)
box_length_y = 53.7829
box_length_z = 53.7829

# 沿 x 方向的分箱宽度 (Å)，比如文章里的 0.05
bin_width = 0.05

######################################
# 加载xyz
######################################

def load_xyz_for_density(filename):
    frames = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            num_atoms = int(line.strip())
            _ = f.readline()  # 跳过注释行
            atoms = []
            for _ in range(num_atoms):
                parts = f.readline().strip().split()
                x = float(parts[1])
                atoms.append(x)
            frames.append(np.array(atoms))
    return frames

######################################
# 画图函数
######################################

def plot_fine_scale_density_x(
    frames,
    bin_width,
    box_length_y,
    box_length_z,
    output_dir,
    output_name
):
    # 合并所有帧
    all_x = np.concatenate(frames)
    print(f"[INFO] Total x-coordinates from all frames: {len(all_x)}")

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

    # 画
    plt.figure(figsize=(8, 5))
    plt.plot(centers, number_density, label='Fine scale density (x direction)')
    plt.xlabel('x (Å)')
    plt.ylabel('Number density (atoms/Å³)')
    plt.title('Fine-scale density profile along x')
    plt.legend()
    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SUCCESS] Plot saved to: {out_path}")

######################################
# 主流程
######################################

if __name__ == "__main__":
    print("[START] Loading xyz file...")
    frames = load_xyz_for_density(xyz_file)
    print("[INFO] Number of frames loaded:", len(frames))

    print("[START] Generating density profile...")
    plot_fine_scale_density_x(
        frames,
        bin_width=bin_width,
        box_length_y=box_length_y,
        box_length_z=box_length_z,
        output_dir=output_dir,
        output_name=output_image_name
    )
    print("[DONE] All finished!")
