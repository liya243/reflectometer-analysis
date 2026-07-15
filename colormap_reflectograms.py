import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from raw_data import read_reflectograms, sellmeier_n


LIGHT_SPEED_M_PER_S = 299792458.0


def distance_axis_from_sampling_rate(n_samples, sampling_rate_hz, wavelength_um=1.55):
    n_eff = float(sellmeier_n(wavelength_um))
    distance_step_m = LIGHT_SPEED_M_PER_S / (2.0 * n_eff * float(sampling_rate_hz))
    return np.arange(n_samples, dtype=np.float64) * distance_step_m, distance_step_m


def select_parity_subset(trace_matrix, parity):
    data = np.asarray(trace_matrix)
    global_indices = np.arange(data.shape[0], dtype=np.int64)
    if parity == "all":
        return data, global_indices
    if parity == "even":
        mask = (global_indices % 2) == 0
    elif parity == "odd":
        mask = (global_indices % 2) == 1
    else:
        raise ValueError(f"Unsupported parity: {parity}")
    return data[mask], global_indices[mask]


def save_png_figure(fig, png_path):
    png_path = Path(png_path)
    fig.savefig(png_path, dpi=200)
    return png_path


def main():
    parser = argparse.ArgumentParser(
        description="Построить цветную карту рефлектограмм из .dat-файла."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Каталог для выходного изображения",
    )
    parser.add_argument(
        "--trace-stride",
        type=int,
        default=1,
        help="Оставлять каждую N-ю рефлектограмму по вертикальной оси",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Оставлять каждый N-й отсчёт по горизонтальной оси",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="Нижний процентиль для ограничения цветовой шкалы",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.0,
        help="Верхний процентиль для ограничения цветовой шкалы",
    )
    parser.add_argument(
        "--scan-rate",
        type=float,
        default=None,
        help="Необязательная частота записи рефлектограмм в Hz",
    )
    parser.add_argument(
        "--length-m",
        type=float,
        default=None,
        help="Необязательный горизонтальный размер в метрах; если не задан, ось расстояний считается по sampling_rate",
    )
    parser.add_argument(
        "--ignore-first-meters",
        type=float,
        default=0.0,
        help="Игнорировать столько метров от начала каждой рефлектограммы",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=None,
        help="Необязательная максимальная координата в метрах",
    )
    parser.add_argument(
        "--subtract-time-mean",
        action="store_true",
        help="Вычесть среднюю по времени трассу перед построением, чтобы подчеркнуть модуляцию",
    )
    parser.add_argument(
        "--parity",
        choices=["all", "even", "odd"],
        default="all",
        help="Использовать все рефлектограммы или только одну чётность",
    )
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float32)
    if args.length_m is None:
        full_x_axis, distance_step_m = distance_axis_from_sampling_rate(
            result["real_segment_size"],
            result["sampling_rate"],
        )
    else:
        full_x_axis = np.linspace(0.0, float(args.length_m), result["real_segment_size"], dtype=np.float64)
        distance_step_m = float(np.median(np.diff(full_x_axis))) if full_x_axis.size > 1 else float("nan")

    x_mask = full_x_axis >= float(args.ignore_first_meters)
    if args.max_distance_m is not None:
        x_mask &= full_x_axis <= float(args.max_distance_m)
    if not np.any(x_mask):
        raise ValueError("Selected distance range removes the full reflectogram")

    data = data[:, x_mask]
    x_axis = full_x_axis[x_mask]
    data, selected_global_indices = select_parity_subset(data, args.parity)
    if args.subtract_time_mean:
        data = data - data.mean(axis=0, keepdims=True)
    data_view = data[:: args.trace_stride, :: args.sample_stride]
    x_axis_view = x_axis[:: args.sample_stride]
    time_axis_s = selected_global_indices[:: args.trace_stride].astype(np.float64) / result["scan_rate"]

    vmin = float(np.percentile(data_view, args.lower_percentile))
    vmax = float(np.percentile(data_view, args.upper_percentile))
    x_label = "Distance (m)"

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    im = ax.imshow(
        data_view,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[x_axis_view[0], x_axis_view[-1], time_axis_s[0], time_axis_s[-1]],
    )
    title = dat_path.name
    if args.parity != "all":
        title += f" ({args.parity})"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Время (s)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Signal")

    suffix_tag = ""
    if args.parity != "all":
        suffix_tag += f"_{args.parity}"
    if args.ignore_first_meters > 0.0:
        suffix_tag += f"_ignore_first_{int(round(args.ignore_first_meters))}m"
    if args.max_distance_m is not None:
        suffix_tag += f"_to_{int(round(args.max_distance_m))}m"
    if args.subtract_time_mean:
        suffix_tag += "_demeaned_over_time"
    output_path = output_dir / f"{dat_path.stem}_colormap{suffix_tag}.png"
    saved_png_path = save_png_figure(fig, output_path)
    plt.close(fig)

    print(f"file: {dat_path}")
    print(f"shape: {result['data'].shape}")
    print(f"stored_scan_rate_hz: {result['stored_scan_rate']}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"parity: {args.parity}")
    print(f"duration_s: {(selected_global_indices[-1] - selected_global_indices[0]) / result['scan_rate']:.6f}")
    print(f"distance_step_m: {distance_step_m:.6f}")
    if args.length_m is not None:
        print(f"length_m_override: {args.length_m}")
    print(f"ignore_first_meters: {args.ignore_first_meters}")
    if args.max_distance_m is not None:
        print(f"max_distance_m: {args.max_distance_m}")
    print(f"subtract_time_mean: {args.subtract_time_mean}")
    print(f"trace_stride: {args.trace_stride}")
    print(f"sample_stride: {args.sample_stride}")
    print(f"plotted_shape: {data_view.shape}")
    print(f"plotted_distance_start_m: {x_axis_view[0]:.6f}")
    print(f"plotted_distance_end_m: {x_axis_view[-1]:.6f}")
    print(f"vmin: {vmin:.6f}")
    print(f"vmax: {vmax:.6f}")
    print(f"plot_png_saved_to: {saved_png_path}")


if __name__ == "__main__":
    main()
