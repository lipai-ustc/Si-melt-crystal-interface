import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def plot_entropy_enthalpy_single_face(centers_entropy, entropy_values, centers_enthalpy, enthalpy_values,
                                      face_label, color, out_path, x_min_plot=None, x_max_plot=None):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    fs = 23
    ax1.set_xlabel("z (Å)", fontsize=fs)
    ax1.set_ylabel("Entropy (eV/atom)", fontsize=fs)
    ax2.set_ylabel("Enthalpy (eV/atom)", fontsize=fs)

    # 选取 z 范围
    if x_min_plot is not None:
        mask1 = centers_entropy >= x_min_plot
        mask2 = centers_enthalpy >= x_min_plot
    else:
        mask1 = mask2 = slice(None)
    if x_max_plot is not None:
        mask1 = mask1 & (centers_entropy <= x_max_plot)
        mask2 = mask2 & (centers_enthalpy <= x_max_plot)

    # Plot
    ax1.plot(centers_entropy[mask1], entropy_values[mask1], color=color, lw=2, label=f"Entropy {face_label}", alpha=0.9, ls='-')
    ax2.plot(centers_enthalpy[mask2], enthalpy_values[mask2], color=color, lw=2, label=f"Enthalpy {face_label}", alpha=0.9, ls='--')

    # 美化
    ax1.axvline(0, color='gray', ls='--', lw=1)
    ax1.axvline(-10, color='gray', ls='--', lw=1)
    ax1.yaxis.set_major_locator(MultipleLocator(0.15))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax2.yaxis.set_major_locator(MultipleLocator(0.15))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(3))

    ax1.tick_params(axis='both', labelsize=fs)
    ax2.tick_params(axis='y', labelsize=fs)
    if x_min_plot is not None and x_max_plot is not None:
        ax1.set_xlim(x_min_plot, x_max_plot)
    ax1.set_ylim(-1.7, -1.2)
    ax2.set_ylim(-9.85, -9.25)

    # 图例
    ax1.legend([f"{face_label} Entropy", f"{face_label} Enthalpy"], fontsize=fs - 5, loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"[SUCCESS] Saved: {out_path}")

def main():
    base_dir = r"E:\free_energy\0all_new_pot"
    output_dir = os.path.join(base_dir, "separate_face_plots")
    os.makedirs(output_dir, exist_ok=True)

    rel_range = 20
    faces = ["100", "110", "111"]
    colors = {"100": "C0", "110": "C1", "111": "C2"}

    for face in faces:
        entropy_path = os.path.join(base_dir, face, "entropy", f"{face}.csv")
        enthalpy_path = os.path.join(base_dir, face, "enthalpy", f"{face}.csv")

        if not os.path.isfile(entropy_path):
            print(f"[WARN] Entropy CSV not found: {entropy_path}")
            continue
        if not os.path.isfile(enthalpy_path):
            print(f"[WARN] Enthalpy CSV not found: {enthalpy_path}")
            continue

        # 读取数据
        entropy_data = np.loadtxt(entropy_path, delimiter=',', skiprows=1)
        enthalpy_data = np.loadtxt(enthalpy_path, delimiter=',', skiprows=1)

        entropy_centers = entropy_data[:, 0]
        entropy_values = entropy_data[:, 1]
        enthalpy_centers = enthalpy_data[:, 0]
        enthalpy_values = enthalpy_data[:, 1]

        out_file = os.path.join(output_dir, f"{face}_entropy_enthalpy_plot.png")
        plot_entropy_enthalpy_single_face(
            entropy_centers, entropy_values,
            enthalpy_centers, enthalpy_values,
            face_label=face,
            color=colors[face],
            out_path=out_file,
            x_min_plot=-rel_range,
            x_max_plot=rel_range
        )

if __name__ == "__main__":
    main()
