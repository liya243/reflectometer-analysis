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
    return np.arange(n_samples, dtype=np.float64) * distance_step_m


def render_colormap(dat_path, data_view, distance_view_m, time_view_s, title, output_path):
    vmin = float(np.percentile(data_view, 1.0))
    vmax = float(np.percentile(data_view, 99.0))

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    im = ax.imshow(
        data_view,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[distance_view_m[0], distance_view_m[-1], time_view_s[0], time_view_s[-1]],
    )
    ax.set_title(title)
    ax.set_xlabel("Расстояние (m)")
    ax.set_ylabel("Время (s)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Signal relative to zero level")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return vmin, vmax


def main():
    parser = argparse.ArgumentParser(
        description="Построить цветные карты импульсов в сервисной зоне отдельно для чётных и нечётных рефлектограмм."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Каталог для выходных изображений",
    )
    parser.add_argument(
        "--scan-rate",
        type=float,
        default=None,
        help="Необязательная частота записи рефлектограмм в Hz",
    )
    parser.add_argument(
        "--distance-min-m",
        type=float,
        default=70.0,
        help="Начало окна импульса в сервисной зоне, в метрах",
    )
    parser.add_argument(
        "--distance-max-m",
        type=float,
        default=90.0,
        help="Конец окна импульса в сервисной зоне, в метрах",
    )
    parser.add_argument(
        "--zero-level-m",
        type=float,
        default=70.0,
        help="Координата в метрах, используемая как нулевой уровень каждой рефлектограммы",
    )
    parser.add_argument(
        "--subtract-time-mean",
        action="store_true",
        help="После коррекции нулевого уровня вычесть среднюю по времени импульсную трассу внутри каждой чётности",
    )
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float64)
    distance_axis_m = distance_axis_from_sampling_rate(result["real_segment_size"], result["sampling_rate"])

    pulse_mask = (distance_axis_m >= float(args.distance_min_m)) & (distance_axis_m <= float(args.distance_max_m))
    if not np.any(pulse_mask):
        raise ValueError("Requested pulse window is empty")

    zero_index = int(np.argmin(np.abs(distance_axis_m - float(args.zero_level_m))))
    zero_level_actual_m = float(distance_axis_m[zero_index])
    zeroed = data - data[:, [zero_index]]

    pulse_data = zeroed[:, pulse_mask]
    pulse_distance_m = distance_axis_m[pulse_mask]

    parity_specs = [
        ("even", pulse_data[0::2], np.arange(0, result["refls_count"], 2, dtype=np.int64)),
        ("odd", pulse_data[1::2], np.arange(1, result["refls_count"], 2, dtype=np.int64)),
    ]

    for parity_name, parity_data, global_indices in parity_specs:
        view = parity_data.copy()
        if args.subtract_time_mean:
            view -= view.mean(axis=0, keepdims=True)
        time_axis_s = global_indices.astype(np.float64) / result["scan_rate"]
        suffix = f"{parity_name}_{int(round(args.distance_min_m))}_{int(round(args.distance_max_m))}m_zero_{int(round(args.zero_level_m))}m"
        if args.subtract_time_mean:
            suffix += "_demeaned_over_time"
        output_path = output_dir / f"{dat_path.stem}_pulse_colormap_{suffix}.png"
        vmin, vmax = render_colormap(
            dat_path=dat_path,
            data_view=view,
            distance_view_m=pulse_distance_m,
            time_view_s=time_axis_s,
            title=f"{dat_path.name} pulse colormap ({parity_name}, {args.distance_min_m:.0f}-{args.distance_max_m:.0f} m)",
            output_path=output_path,
        )
        print(f"parity: {parity_name}")
        print(f"zero_level_actual_m: {zero_level_actual_m:.6f}")
        print(f"trace_count: {view.shape[0]}")
        print(f"distance_start_m: {pulse_distance_m[0]:.6f}")
        print(f"distance_end_m: {pulse_distance_m[-1]:.6f}")
        print(f"vmin: {vmin:.6f}")
        print(f"vmax: {vmax:.6f}")
        print(f"plot_png_saved_to: {output_path}")


if __name__ == "__main__":
    main()
