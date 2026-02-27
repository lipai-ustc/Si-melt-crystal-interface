def parse_positions_xyz(filename):
    all_positions = []
    all_lattices = []
    all_origins = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith('ITEM: TIMESTEP'):
            num_atoms = int(lines[i + 3])

            xlo, xhi = map(float, lines[i + 5].split())
            ylo, yhi = map(float, lines[i + 6].split())
            zlo, zhi = map(float, lines[i + 7].split())

            a = xhi - xlo
            e = yhi - ylo
            i_length = zhi - zlo

            lattice_line = f'{a} 0.0 0.0 0.0 {e} 0.0 0.0 0.0 {i_length}'
            origin_line = f'{xlo} {ylo} {zlo}'

            frame_positions = []
            data_start = i + 9
            for j in range(data_start, data_start + num_atoms):
                parts = lines[j].split()
                x, y, z = parts[2], parts[3], parts[4]
                frame_positions.append((x, y, z))

            all_positions.append(frame_positions)
            all_lattices.append(lattice_line)
            all_origins.append(origin_line)

            i = data_start + num_atoms
        else:
            i += 1

    return all_positions, all_lattices, all_origins


def write_extendxyz_file(output_file, positions, lattices, origins):
    with open(output_file, 'w') as f:
        for frame_idx, (pos_frame, lattice, origin) in enumerate(zip(positions, lattices, origins), 1):
            f.write(f"{len(pos_frame)}\n")
            f.write(f'Lattice="{lattice}" Origin="{origin}" Properties=species:S:1:pos:R:3 Frame={frame_idx}\n')
            for x, y, z in pos_frame:
                f.write(f"Si {x} {y} {z}\n")


# 文件路径
input_file = r"E:\free_energy\0all_new_pot\111\cal_result\energy.xyz"
output_file = r"E:\free_energy\0all_new_pot\111\msd\energy_extend.xyz"

# 处理与写出
positions, lattices, origins = parse_positions_xyz(input_file)
write_extendxyz_file(output_file, positions, lattices, origins)
