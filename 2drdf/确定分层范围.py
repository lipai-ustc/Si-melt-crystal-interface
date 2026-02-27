import numpy as np
'''100:layers_relative = {
    'D': (-5.3, -4),
    'C': (-4, -2.675),
    'B': (-2.675, -1.225),
    'A': (-1.225, 0.175),
    'liquid':(-11, -9.6)
}'''
'''100:layers_relative = {
    'C': (-2.675, -1.225),
    'D': (-1.225, 0.175),
    'E': (0.175, 1.525),
    'F': (1.525, 2.975)
}'''
'''100:layers_relative = {
    'Y': (-8.2, -6.8),
    'Z': (-6.8, -5.4)
    'A': (-5.3, -4),
    'B': (-4, -2.675)

}'''
'''100:layers_relative = {
    'W': (-11, -9.6),
    'X': (-9.6, -8.2),
    'Y': (-8.2, -6.8),
    'Z': (-6.8, -5.4)
}'''
'''110:layers_relative = {
    'A': (-8.125, -6.225),
    'B': (-6.225, -4.325),
    'C': (-4.325, -2.425),
    'D': (-2.425, -0.575)
}'''
'''111:layers_relative = {
    'A': (-10.125, -7.275),
    'B': (-7.275, -4.625),
    'C': (-4.625, -1.525),
    'D': (-1.525, 1.725)
}'''
# 你的界面位置文件
interface_file = r"E:\free_energy\all_new_pot\111\cal-result\interface_lq6-bins13.txt"
output_file = r"E:\free_energy\all_new_pot\111\2drdf\ABCDE_positions.txt"

# 相对范围
layers_relative = {
    'A': (-1.525, 1.725),
    'B': (-4.625, -1.525),
    'C': (-7.275, -4.625),
    'D': (-10.125, -7.275),
    'E': (-15, -12.15)
}
# 读取界面位置
def load_interface_positions(filename):
    positions = []
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            if line.strip():
                parts = line.strip().split()
                positions.append(float(parts[0]))
    return np.array(positions)

interface_positions = load_interface_positions(interface_file)
print(f"[INFO] Loaded {len(interface_positions)} interface positions.")

# 写到文件
with open(output_file, 'w') as fout:
    for frame_index, interface_pos in enumerate(interface_positions):
        fout.write(f"Frame {frame_index}\n")
        for layer_name, (rel_min, rel_max) in layers_relative.items():
            abs_min = interface_pos + rel_min
            abs_max = interface_pos + rel_max
            fout.write(f"{layer_name} {abs_min:.6f} {abs_max:.6f}\n")
        fout.write("\n")


print(f"[SUCCESS] Wrote layer positions to: {output_file}")
