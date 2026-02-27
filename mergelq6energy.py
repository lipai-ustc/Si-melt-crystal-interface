# 解析energy文件，同时读取energy、entropy和box信息
def parse_energy_and_entropy_xyz(filename):
    energy_frames = []
    entropy_frames = []
    box_frames = []    # 存储Lattice line
    origin_frames = [] # 存储Origin line

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith('ITEM: TIMESTEP'):
            num_atoms = int(lines[i + 3])

            # 正确偏移量
            xlo, xhi = map(float, lines[i + 5].split())
            ylo, yhi = map(float, lines[i + 6].split())
            zlo, zhi = map(float, lines[i + 7].split())

            # 转换成Lattice格式
            a = xhi - xlo
            e = yhi - ylo
            i_length = zhi - zlo

            lattice_line = f'{a} 0.0 0.0 0.0 {e} 0.0 0.0 0.0 {i_length}'
            origin_line = f'{xlo} {ylo} {zlo}'

            box_frames.append(lattice_line)
            origin_frames.append(origin_line)

            # 读取energy和entropy
            data_start = i + 9  # ITEM: ATOMS 行在 i+8，数据从 i+9 开始
            frame_energy = []
            frame_entropy = []
            for j in range(data_start, data_start + num_atoms):
                parts = lines[j].split()
                energy = float(parts[5])    # v_E_total在第6列
                entropy = float(parts[6])   # c_Entropy在第7列
                frame_energy.append(energy)
                frame_entropy.append(entropy)

            energy_frames.append(frame_energy)
            entropy_frames.append(frame_entropy)

            i = data_start + num_atoms
        else:
            i += 1

    return energy_frames, entropy_frames, box_frames, origin_frames

# 解析lq6文件
def parse_lq6_xyz(filename):
    frames = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        lattice_line = lines[i + 1].strip()  # 这里虽然读取了，但后面不用 lattice 了
        data_start = i + 2
        frame_data = []
        for j in range(data_start, data_start + num_atoms):
            parts = lines[j].split()
            x, y, z = parts[1], parts[2], parts[3]  # 保持字符串格式
            lq6 = float(parts[4])
            frame_data.append((x, y, z, lq6))
        frames.append(frame_data)
        i = data_start + num_atoms
    return frames

# 写merge的tra.xyz文件，box来自energy.xyz，Origin加上
def write_tra_xyz(output_filename, energy_frames, entropy_frames, lq6_frames, box_frames, origin_frames):
    with open(output_filename, 'w') as f:
        for frame_index, (e_frame, ent_frame, lq6_frame, lattice, origin) in enumerate(
                zip(energy_frames, entropy_frames, lq6_frames, box_frames, origin_frames), 1):
            num_atoms = len(e_frame)
            f.write(f"{num_atoms}\n")
            f.write(
                f'Lattice="{lattice}" Origin="{origin}" Properties=species:S:1:pos:R:3:lq6:R:1:entropy:R:1:energy:R:1 Frame={frame_index}\n')
            for i in range(num_atoms):
                x, y, z = lq6_frame[i][0], lq6_frame[i][1], lq6_frame[i][2]
                lq6 = lq6_frame[i][3]
                entropy = ent_frame[i]
                energy = e_frame[i]
                f.write(f"Si {x} {y} {z} {lq6} {entropy} {energy}\n")

# 使用新代码生成merge.xyz
energy_file = r"E:\free_energy\400-newpot\110\energy.xyz"
lq6_file = r"E:\free_energy\400-newpot\110\lq6-2.xyz"
output_file = r"E:\free_energy\400-newpot\110\merge.xyz"

energy_data, entropy_data, box_data, origin_data = parse_energy_and_entropy_xyz(energy_file)
lq6_data = parse_lq6_xyz(lq6_file)

# 检查帧数是否匹配
if len(energy_data) != len(lq6_data):
    raise ValueError(f"Error: energy file has {len(energy_data)} frames, but lq6 file has {len(lq6_data)} frames! Please check.")

# 写文件
write_tra_xyz(output_file, energy_data, entropy_data, lq6_data, box_data, origin_data)
