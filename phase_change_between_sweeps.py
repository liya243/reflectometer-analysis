import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

from analysis_output_utils import matlab_safe_stem


def wrap_phase(values):
    values = np.asarray(values, dtype=np.float64)
    return np.angle(np.exp(1j * values))


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)
    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Phase change between sweeps');
imagesc(data.chain_distance_m, data.sweep_pair_index, data.phase_change_wrapped);
axis xy; colorbar; xlabel('Distance (m)'); ylabel('Sweep pair index');
title('arg(E_{{n+1}}) - arg(E_n) wrapped to [-pi, pi]');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return {"mat": mat_path, "script": script_path}


def main():
    parser = argparse.ArgumentParser(
        description="Построить изменения фазы от свипа к свипу по результатам пакетного восстановления комплексных амплитуд."
    )
    parser.add_argument("mat_path", help="Путь к *_complex_amplitudes_over_sweeps_matlab_data.mat")
    parser.add_argument("--output-dir", default=None, help="Каталог для выходных файлов; по умолчанию каталог MAT-файла")
    args = parser.parse_args()

    mat_path = Path(args.mat_path)
    output_dir = Path(args.output_dir) if args.output_dir is not None else mat_path.parent
    output_dir.mkdir(exist_ok=True)

    data = loadmat(mat_path)
    chain_distance_m = np.asarray(data["chain_distance_m"], dtype=np.float64).reshape(-1)
    sweep_index = np.asarray(data["sweep_index"], dtype=np.int64).reshape(-1)
    phase_matrix = np.asarray(data["E_phase_over_sweeps"], dtype=np.float64)
    if phase_matrix.shape[0] < 2:
        raise ValueError("Need at least two sweeps to compute phase changes")

    phase_change = np.diff(phase_matrix, axis=0)
    phase_change_wrapped = wrap_phase(phase_change)
    sweep_pair_index = sweep_index[:-1]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    im = ax.imshow(
        phase_change_wrapped,
        aspect="auto",
        origin="lower",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        extent=[chain_distance_m[0], chain_distance_m[-1], sweep_pair_index[0], sweep_pair_index[-1]],
    )
    ax.set_xlabel("Расстояние (m)")
    ax.set_ylabel("Индекс пары свипов")
    ax.set_title("Wrapped-изменение фазы между соседними свипами")
    fig.colorbar(im, ax=ax, label="Delta phase (rad)")

    stem = mat_path.name.replace("_complex_amplitudes_over_sweeps_matlab_data.mat", "")
    png_path = output_dir / f"{stem}_complex_amplitude_phase_change_between_sweeps.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=stem,
        suffix_tag="complex_amplitude_phase_change_between_sweeps",
        payload={
            "chain_distance_m": chain_distance_m[:, None],
            "sweep_pair_index": sweep_pair_index[:, None],
            "phase_change": phase_change,
            "phase_change_wrapped": phase_change_wrapped,
        },
    )

    print(f"mat_file: {mat_path}")
    print(f"sweep_count: {phase_matrix.shape[0]}")
    print(f"coordinate_count: {phase_matrix.shape[1]}")
    print(f"phase_change_png_saved_to: {png_path}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
