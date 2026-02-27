import os

# 玻尔兹曼常数 (eV/K) 和温度
k_B = 8.617333262e-5  # 单位：eV/K
T = 1844  # 温度（K）


def parse_merge_xyz(filepath):
    """读取 merge.xyz 文件，提取每帧的信息：Lattice, Origin, Frame, 坐标, 能量"""
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        header_line = lines[i + 1].strip()
        atoms = lines[i + 2: i + 2 + num_atoms]

        # 解析 Lattice/Origin 行
        lattice_origin = header_line.split("Properties=")[0].strip()
        frame_info = ""
        if "Frame=" in header_line:
            frame_info = "Frame=" + header_line.split("Frame=")[1].strip()

        species_xyz = [line.strip().split()[:4] for line in atoms]
        energies = [float(line.strip().split()[-1]) for line in atoms]

        frames.append({
            'num_atoms': num_atoms,
            'lattice_origin': lattice_origin,
            'frame_info': frame_info,
            'species_xyz': species_xyz,
            'energies': energies
        })

        i += 2 + num_atoms

    return frames


def parse_realentropy_xyz(filepath):
    """读取 realentropy.xyz 文件，提取每帧的熵"""
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        atoms = lines[i + 2: i + 2 + num_atoms]
        entropies = [float(line.strip().split()[-1]) for line in atoms]

        frames.append({
            'num_atoms': num_atoms,
            'entropies': entropies
        })

        i += 2 + num_atoms

    return frames


def compute_gibbs_energy(energy, entropy):
    return energy - entropy * k_B * T


def write_gibbs_xyz(output_path, merge_frames, entropy_frames):
    with open(output_path, 'w') as f:
        for m_frame, s_frame in zip(merge_frames, entropy_frames):
            if m_frame['num_atoms'] != s_frame['num_atoms']:
                raise ValueError("每帧原子数不匹配")

            f.write(f"{m_frame['num_atoms']}\n")
            # 保留 Lattice + Origin，替换 Properties 为：species + gibbs
            new_properties = 'Properties=species:S:1:pos:R:3:gibbs:R:1'
            header = f"{m_frame['lattice_origin']} {new_properties}"
            if m_frame['frame_info']:
                header += f" {m_frame['frame_info']}"
            f.write(header + "\n")

            for atom, E, S in zip(m_frame['species_xyz'], m_frame['energies'], s_frame['entropies']):
                G = compute_gibbs_energy(E, S)
                f.write(f"{' '.join(atom)} {G:.6f}\n")

def main():
    merge_file = r"E:\free_energy\0all_new_pot\100\cal_result\merge.xyz"
    entropy_file = r"E:\free_energy\0_all_newpot-2\100\entropy\realentropy.xyz"
    output_file = r"E:\free_energy\0_all_newpot-2\100\gibbs\gibbs_free_energy.xyz"

    if not os.path.exists(merge_file) or not os.path.exists(entropy_file):
        print("❌ 输入文件不存在，请检查路径。")
        return

    merge_frames = parse_merge_xyz(merge_file)
    entropy_frames = parse_realentropy_xyz(entropy_file)
    write_gibbs_xyz(output_file, merge_frames, entropy_frames)

    print(f"✅ Gibbs 自由能计算完成，输出写入：{output_file}")

if __name__ == "__main__":
    main()
