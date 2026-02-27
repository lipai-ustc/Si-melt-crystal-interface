from ase.io import read, write
from ase import Atoms

input_file = r"E:\free_energy\0all_new_pot\100\cal_result\merge.xyz"
output_file = r"E:\free_energy\0all_new_pot\100\cal_result\atoms.xyz"

# 读取所有帧
all_frames = read(input_file, index=":")

clean_frames = []

for frame in all_frames:
    # 只保留符号、位置、晶格和PBC
    clean_frame = Atoms(
        symbols=frame.get_chemical_symbols(),
        positions=frame.get_positions(),
        cell=frame.get_cell(),
        pbc=frame.get_pbc(),
    )
    clean_frames.append(clean_frame)

# 写出新的 extended xyz 文件
write(output_file, clean_frames, format="extxyz")

print(f"转换完成，共处理 {len(clean_frames)} 帧，结果保存在 {output_file}")
