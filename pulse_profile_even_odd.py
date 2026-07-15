import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from analysis_output_utils import matlab_safe_stem
from raw_data import read_reflectograms
from reflectometer_utils import distance_axis_from_sampling_rate, subtract_trace_baseline_from_tail


def save_png_figure(fig, png_path):
    png_path = Path(png_path)
    fig.savefig(png_path, dpi=200)
    return png_path


def save_matlab_bundle(output_dir, stem, suffix_tag, distance_m, even_profile, odd_profile, zero_level_m):
    output_dir = Path(output_dir)
    data_mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"

    savemat(
        data_mat_path,
        {
            "distance_m": np.asarray(distance_m, dtype=np.float64),
            "even_profile": np.asarray(even_profile, dtype=np.float64),
            "odd_profile": np.asarray(odd_profile, dtype=np.float64),
            "zero_level_m": np.array([[zero_level_m]], dtype=np.float64),
        },
    )

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{data_mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Even/odd pulse profile');
plot(data.distance_m, data.even_profile, 'LineWidth', 1.6);
hold on;
plot(data.distance_m, data.odd_profile, 'LineWidth', 1.6);
grid on;
xlabel('Distance (m)');
ylabel('Signal relative to zero level');
title(sprintf('Форма импульса, нулевой уровень в %.3f m', data.zero_level_m));
legend('Even reflectograms', 'Odd reflectograms', 'Location', 'best');

% Optional: save native MATLAB figure file after opening it.
% savefig(f1, fullfile(this_dir, '{stem}_{suffix_tag}.fig'));
"""
    script_path.write_text(script_text, encoding="utf-8")

    return {
        "mat": data_mat_path,
        "script": script_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Построить средние формы чётного и нечётного импульса по сервисной зоне."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Каталог для выходных файлов",
    )
    parser.add_argument(
        "--scan-rate",
        type=float,
        default=None,
        help="Необязательная частота записи рефлектограмм в Hz",
    )
    parser.add_argument(
        "--pulse-z-min",
        type=float,
        default=70.0,
        help="Начало окна импульса в сервисной зоне, в метрах",
    )
    parser.add_argument(
        "--pulse-z-max",
        type=float,
        default=90.0,
        help="Конец окна импульса в сервисной зоне, в метрах",
    )
    parser.add_argument(
        "--zero-level-z",
        type=float,
        default=70.0,
        help="Координата в метрах, используемая как нулевой уровень формы импульса",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Оставить только координаты, где средний чётный или нечётный профиль импульса выше нуля",
    )
    parser.add_argument(
        "--baseline-tail-m",
        type=float,
        default=None,
        help="Вычесть базовый уровень каждой трассы, оцененный по последним N метрам",
    )
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float64)
    distance_axis_m, distance_step_m, _ = distance_axis_from_sampling_rate(
        result["real_segment_size"],
        result["sampling_rate"],
    )
    baseline_start_m = None
    baseline_end_m = None
    if args.baseline_tail_m is not None:
        data, _, _, baseline_start_m, baseline_end_m = subtract_trace_baseline_from_tail(
            data,
            distance_axis_m,
            args.baseline_tail_m,
        )

    pulse_mask = (distance_axis_m >= float(args.pulse_z_min)) & (distance_axis_m <= float(args.pulse_z_max))
    if not np.any(pulse_mask):
        raise ValueError("Pulse window is empty")

    zero_index = int(np.argmin(np.abs(distance_axis_m - float(args.zero_level_z))))
    zero_level_m = float(distance_axis_m[zero_index])
    baseline = data[:, [zero_index]]
    zeroed = data - baseline

    pulse_distance_m = distance_axis_m[pulse_mask]
    pulse_zeroed = zeroed[:, pulse_mask]
    even_profile = np.mean(pulse_zeroed[0::2], axis=0)
    odd_profile = np.mean(pulse_zeroed[1::2], axis=0)
    if args.positive_only:
        positive_mask = (even_profile > 0.0) | (odd_profile > 0.0)
        if not np.any(positive_mask):
            raise ValueError("positive_only removed the full pulse support")
        pulse_distance_m = pulse_distance_m[positive_mask]
        even_profile = even_profile[positive_mask]
        odd_profile = odd_profile[positive_mask]
    else:
        positive_mask = np.ones(pulse_distance_m.size, dtype=bool)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(pulse_distance_m, even_profile, marker="o", linewidth=1.6, markersize=4.0, label="Чётные рефлектограммы")
    ax.plot(pulse_distance_m, odd_profile, marker="o", linewidth=1.6, markersize=4.0, label="Нечётные рефлектограммы")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    if pulse_distance_m[0] <= zero_level_m <= pulse_distance_m[-1]:
        ax.axvline(zero_level_m, color="#666666", linestyle=":", linewidth=0.8, label=f"Zero level {zero_level_m:.3f} m")
    ax.set_xlabel("Расстояние (m)")
    ax.set_ylabel("Сигнал относительно нулевого уровня")
    ax.set_title("Форма импульса в сервисной зоне по чётности")
    ax.grid(alpha=0.25)
    ax.legend()

    suffix_tag = (
        f"pulse_profile_even_odd_{int(round(args.pulse_z_min))}_{int(round(args.pulse_z_max))}m"
        f"_zero_{int(round(args.zero_level_z))}m"
    )
    if args.positive_only:
        suffix_tag += "_positive_only"
    if args.baseline_tail_m is not None:
        suffix_tag += f"_baseline_tail_{int(round(args.baseline_tail_m))}m"
    png_path = output_dir / f"{dat_path.stem}_{suffix_tag}.png"
    saved_png_path = save_png_figure(fig, png_path)
    plt.close(fig)

    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag=suffix_tag,
        distance_m=pulse_distance_m,
        even_profile=even_profile,
        odd_profile=odd_profile,
        zero_level_m=zero_level_m,
    )

    csv_path = output_dir / f"{dat_path.stem}_{suffix_tag}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("distance_m,even_profile,odd_profile\n")
        for z_m, even_value, odd_value in zip(pulse_distance_m, even_profile, odd_profile):
            fout.write(f"{z_m:.10f},{even_value:.10f},{odd_value:.10f}\n")

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"distance_step_m: {distance_step_m:.6f}")
    print(f"pulse_z_min_m: {args.pulse_z_min}")
    print(f"pulse_z_max_m: {args.pulse_z_max}")
    print(f"zero_level_requested_m: {args.zero_level_z}")
    print(f"zero_level_actual_m: {zero_level_m:.6f}")
    print(f"zero_level_index: {zero_index}")
    if args.baseline_tail_m is not None:
        print(f"baseline_tail_m: {args.baseline_tail_m}")
        print(f"baseline_window_start_m: {baseline_start_m:.6f}")
        print(f"baseline_window_end_m: {baseline_end_m:.6f}")
    print(f"positive_only: {args.positive_only}")
    print(f"pulse_samples: {pulse_zeroed.shape[1]}")
    print(f"kept_samples: {pulse_distance_m.size}")
    if args.positive_only:
        print(f"kept_distance_start_m: {pulse_distance_m[0]:.6f}")
        print(f"kept_distance_end_m: {pulse_distance_m[-1]:.6f}")
    print(f"even_trace_count: {pulse_zeroed[0::2].shape[0]}")
    print(f"odd_trace_count: {pulse_zeroed[1::2].shape[0]}")
    print(f"csv_saved_to: {csv_path}")
    print(f"plot_png_saved_to: {saved_png_path}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
