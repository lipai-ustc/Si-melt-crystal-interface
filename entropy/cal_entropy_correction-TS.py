import math

# 常量定义
h = 6.62607015e-34      # 普朗克常数，单位 J·s
k_B = 1.380649e-23      # 玻尔兹曼常数，单位 J/K
T = 1844              # 温度，单位 K（可修改）
mass_si = 4.665e-26     # 硅原子质量（kg）
k_B2 = 8.617333262e-5   # 单位：eV/K
lambda_db = h / math.sqrt(2 * math.pi * mass_si * k_B * T)

def parse_lattice_volume(lattice_line):
    """从 Lattice 行解析盒子体积"""
    parts = lattice_line.split('"')[1].split()
    a, b, c = float(parts[0]), float(parts[4]), float(parts[8])
    return a * b * c

def read_merge_xyz(file_path):
    """读取 merge.xyz 所有帧，返回每帧的初始熵列表和盒子体积"""
    frames = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        lattice_line = lines[i + 1]
        volume = parse_lattice_volume(lattice_line)
        atom_lines = lines[i + 2 : i + 2 + num_atoms]
        entropies = [float(line.split()[5]) for line in atom_lines]
        frames.append({'volume': volume, 'entropies': entropies, 'num_atoms': num_atoms})
        i += 2 + num_atoms

    return frames

def read_atomvolume_xyz(file_path):
    """读取 atomvolume.xyz 所有帧，返回局域密度和原始坐标信息"""
    frames = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        atom_lines = lines[i + 2 : i + 2 + num_atoms]
        densities = []
        base_lines = []
        for line in atom_lines:
            parts = line.split()
            atomic_volume = float(parts[4])
            density = 1e30 / atomic_volume
            densities.append(density)
            base_lines.append(parts[:4])  # id + pos
        frames.append({'densities': densities, 'base_lines': base_lines, 'num_atoms': num_atoms})
        i += 2 + num_atoms

    return frames

def compute_total_entropy(initial_entropy, local_density, average_density):
    """计算总熵和高阶熵"""
    high_order_entropy = initial_entropy / average_density * local_density
    main_entropy = 2.5 - math.log(local_density * lambda_db ** 3)
    total_entropy = high_order_entropy + main_entropy
    # 应用修正公式
    corrected_entropy = total_entropy + 1.53 * high_order_entropy / 1.7169
    return corrected_entropy

def process_frames(merge_file, volume_file, output_file):
    merge_lines = []
    with open(merge_file, 'r') as f:
        merge_lines = f.readlines()

    merge_headers = []
    i = 0
    while i < len(merge_lines):
        num_atoms = int(merge_lines[i].strip())
        header_line = merge_lines[i + 1].strip()
        merge_headers.append({
            "num_atoms": num_atoms,
            "lattice_origin": header_line.split("Properties=")[0].strip(),
            "frame_info": "Frame=" + header_line.split("Frame=")[1].strip() if "Frame=" in header_line else ""
        })
        i += num_atoms + 2

    merge_frames = read_merge_xyz(merge_file)
    volume_frames = read_atomvolume_xyz(volume_file)

    with open(output_file, 'w') as out:
        for m_data, m_frame, v_frame in zip(merge_headers, merge_frames, volume_frames):
            if m_frame['num_atoms'] != v_frame['num_atoms']:
                raise ValueError("merge.xyz 和 atomvolume.xyz 中某一帧的原子数不一致。")

            N = m_frame['num_atoms']
            volume = m_frame['volume']
            avg_density = 1e30 * N / volume

            total_entropies = [
                compute_total_entropy(e, rho, avg_density)
                for e, rho in zip(m_frame['entropies'], v_frame['densities'])
            ]

            # 写入每帧
            out.write(f"{N}\n")
            props = 'Properties=id:I:1:pos:R:3:realentropy:R:1'
            header = f"{m_data['lattice_origin']} {props}"
            if m_data['frame_info']:
                header += f" {m_data['frame_info']}"
            out.write(header + "\n")

            for base, entropy in zip(v_frame['base_lines'], total_entropies):
                # 熵贡献转化为自由能贡献（单位 eV）
                free_energy_entropy = -T * k_B2 * entropy
                out.write(f"{' '.join(base)} {free_energy_entropy:.8f}\n")


# === 主程序入口 ===
if __name__ == "__main__":
    process_frames(
        r"E:\free_energy\0all_new_pot\111\cal_result\merge.xyz",
        r"E:\free_energy\0all_new_pot\111\cal_result\atomvolume.xyz",
        r"E:\free_energy\0_all_newpot-2\111\entropy\realentropy-TS.xyz"#单位是kb
    )
    print("计算完成，结果保存为 realentropy.xyz")
