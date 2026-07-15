import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from raw_data import read_reflectograms
from reflectometer_utils import distance_axis_from_sampling_rate, subtract_trace_baseline_from_tail
from sweep_harmonics_even_odd import detect_reset_times


def select_parity_subset(trace_matrix, parity):
    data = np.asarray(trace_matrix)
    global_indices = np.arange(data.shape[0], dtype=np.int64)
    if parity == "even":
        mask = (global_indices % 2) == 0
    elif parity == "odd":
        mask = (global_indices % 2) == 1
    else:
        raise ValueError(f"Unsupported parity: {parity}")
    return data[mask], global_indices[mask]


def matlab_safe_stem(text):
    safe = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not safe or not safe[0].isalpha():
        safe = f"fig_{safe}"
    return safe


def render_colormap(ax, view, distance_m, time_s, reset_times_s, title, vmin, vmax):
    im = ax.imshow(
        view,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[distance_m[0], distance_m[-1], time_s[0], time_s[-1]],
    )
    for reset_time_s in reset_times_s:
        ax.axhline(float(reset_time_s), color="#D62728", linewidth=0.8, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Расстояние (m)")
    ax.set_ylabel("Время (s)")
    return im


def zoom_limits(reset_times_s, dominant_period_s, default_end_s):
    reset_times_s = np.asarray(reset_times_s, dtype=np.float64)
    if reset_times_s.size == 0 or not np.isfinite(dominant_period_s) or dominant_period_s <= 0.0:
        return 0.0, default_end_s
    start_s = max(0.0, float(reset_times_s[0]) - 0.5 * float(dominant_period_s))
    end_s = min(default_end_s, start_s + 6.0 * float(dominant_period_s))
    if end_s <= start_s:
        end_s = min(default_end_s, start_s + 0.5)
    return start_s, end_s


def tail_zoom_limits(reset_times_s, dominant_period_s, default_end_s, tail_period_count):
    reset_times_s = np.asarray(reset_times_s, dtype=np.float64)
    if reset_times_s.size == 0 or not np.isfinite(dominant_period_s) or dominant_period_s <= 0.0:
        end_s = min(default_end_s, 1.0)
        return max(0.0, end_s - 0.5), end_s
    count = max(1, int(tail_period_count))
    last_idx = reset_times_s.size - 1
    first_idx = max(0, last_idx - count + 1)
    start_s = max(0.0, float(reset_times_s[first_idx]) - 0.5 * float(dominant_period_s))
    end_s = min(float(default_end_s), float(reset_times_s[last_idx]) + 0.5 * float(dominant_period_s))
    if end_s <= start_s:
        end_s = min(float(default_end_s), start_s + max(0.5, float(count) * float(dominant_period_s)))
    return start_s, end_s


def regularize_reset_grid(reset_times_s, dominant_period_s, max_deviation_fraction=0.35):
    reset_times_s = np.asarray(reset_times_s, dtype=np.float64)
    if reset_times_s.size < 2 or not np.isfinite(dominant_period_s) or dominant_period_s <= 0.0:
        return reset_times_s.copy()

    period_s = float(dominant_period_s)
    indices = np.rint((reset_times_s - reset_times_s[0]) / period_s).astype(np.int64)
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    grouped_times = np.zeros(unique_indices.size, dtype=np.float64)
    for idx in range(unique_indices.size):
        grouped_times[idx] = np.median(reset_times_s[inverse == idx])

    intercept_s = float(np.median(grouped_times - unique_indices.astype(np.float64) * period_s))
    regular_times_s = intercept_s + unique_indices.astype(np.float64) * period_s
    keep_mask = np.abs(grouped_times - regular_times_s) <= float(max_deviation_fraction) * period_s
    if np.count_nonzero(keep_mask) >= 2:
        unique_indices = unique_indices[keep_mask]
        grouped_times = grouped_times[keep_mask]
        intercept_s = float(np.median(grouped_times - unique_indices.astype(np.float64) * period_s))
        regular_times_s = intercept_s + unique_indices.astype(np.float64) * period_s
    return regular_times_s


def build_periodic_reset_grid(anchor_time_s, period_s, min_time_s, max_time_s):
    period_s = float(period_s)
    if not np.isfinite(period_s) or period_s <= 0.0:
        raise ValueError("reset period override must be positive")

    anchor_time_s = float(anchor_time_s)
    min_time_s = float(min_time_s)
    max_time_s = float(max_time_s)
    if max_time_s < min_time_s:
        return np.array([], dtype=np.float64)

    first_index = int(np.ceil((min_time_s - anchor_time_s) / period_s))
    last_index = int(np.floor((max_time_s - anchor_time_s) / period_s))
    if last_index < first_index:
        return np.array([], dtype=np.float64)
    indices = np.arange(first_index, last_index + 1, dtype=np.float64)
    return anchor_time_s + indices * period_s


def save_matlab_bundle(output_dir, dat_stem, suffix, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{dat_stem}_{suffix}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{dat_stem}_{suffix}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"

    savemat(mat_path, payload)

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

for k = 1:2
    if k == 1
        traces = data.even_data;
        times = data.even_time_s;
        title_text = 'Even traces';
    else
        traces = data.odd_data;
        times = data.odd_time_s;
        title_text = 'Odd traces';
    end
    figure('Color', 'w', 'Name', ['{dat_stem} ' title_text ' tail zoom']);
    imagesc(data.fiber_distance_m, times, traces);
    axis xy;
    colormap('viridis');
    colorbar;
    hold on;
    for t = data.reset_times_s(:).'
        yline(t, 'r-', 'LineWidth', 0.8);
    end
    ylim([data.tail_zoom_start_s data.tail_zoom_end_s]);
    xlabel('Distance (m)');
    ylabel('Time (s)');
    title([title_text ' tail zoom with regularized reset lines']);
end
"""
    script_path.write_text(script_text, encoding="utf-8")
    return mat_path, script_path


def main():
    parser = argparse.ArgumentParser(
        description="Построить цветные карты чётных/нечётных трасс с вычитанием baseline и найденными концами свипов."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Каталог для выходных файлов")
    parser.add_argument("--scan-rate", type=float, default=None, help="Необязательная частота записи рефлектограмм в Hz")
    parser.add_argument("--fiber-z-min", type=float, default=101.0, help="Начало полезного участка волокна в метрах")
    parser.add_argument("--fiber-z-max", type=float, default=280.0, help="Конец полезного участка волокна в метрах")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Вычесть baseline каждой трассы по последним N метрам")
    parser.add_argument("--trace-stride-full", type=int, default=4, help="Оставлять каждую N-ю трассу на полной карте")
    parser.add_argument("--trace-stride-zoom", type=int, default=1, help="Оставлять каждую N-ю трассу на увеличенной карте")
    parser.add_argument("--sample-stride", type=int, default=1, help="Оставлять каждый N-й отсчёт по оси расстояния")
    parser.add_argument("--rolling-window", type=int, default=64, help="Окно сглаживания детектора сбросов в трассах")
    parser.add_argument("--min-period-s", type=float, default=0.05, help="Минимальный период свипа для детектора сбросов")
    parser.add_argument("--max-period-s", type=float, default=None, help="Максимальный период свипа для детектора сбросов")
    parser.add_argument("--prominence-sigma", type=float, default=3.0, help="Порог детектора сбросов в робастных sigma")
    parser.add_argument("--refine-window-fraction", type=float, default=0.15, help="Окно локального уточнения как доля найденного периода")
    parser.add_argument("--reset-time-shift-ms", type=float, default=0.0, help="Сдвинуть найденные времена сбросов позже на это число ms")
    parser.add_argument(
        "--reset-period-override-ms",
        type=float,
        default=None,
        help="Рисовать линии сбросов с этим фиксированным периодом в ms вместо найденного FFT-периода",
    )
    parser.add_argument(
        "--reset-anchor-time-s",
        type=float,
        default=None,
        help="Опорное время для фиксированной сетки сбросов; сетка продолжается назад и вперёд с заданным периодом",
    )
    parser.add_argument(
        "--shared-reset-detection",
        action="store_true",
        help="Искать времена сбросов один раз по всем трассам и рисовать одинаковые красные линии на even/odd картах",
    )
    parser.add_argument(
        "--max-reset-time-s",
        type=float,
        default=None,
        help="Оставлять только времена сбросов не позже этого абсолютного времени в секундах",
    )
    parser.add_argument(
        "--reset-detection-end-time-s",
        type=float,
        default=None,
        help="Использовать только трассы до этого абсолютного времени при поиске границ периодов модуляции",
    )
    parser.add_argument(
        "--regularize-reset-grid",
        action="store_true",
        help="Спроецировать найденные времена сбросов на регулярную временную сетку с найденным периодом",
    )
    parser.add_argument(
        "--tail-period-count",
        type=int,
        default=8,
        help="Сколько последних периодов модуляции показать на tail zoom",
    )
    parser.add_argument("--lower-percentile", type=float, default=1.0, help="Нижний процентиль для ограничения цветовой шкалы")
    parser.add_argument("--upper-percentile", type=float, default=99.0, help="Верхний процентиль для ограничения цветовой шкалы")
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
    data, baseline_per_trace, _, baseline_start_m, baseline_end_m = subtract_trace_baseline_from_tail(
        data,
        distance_axis_m,
        args.baseline_tail_m,
    )

    fiber_mask = (distance_axis_m >= float(args.fiber_z_min)) & (distance_axis_m <= float(args.fiber_z_max))
    if not np.any(fiber_mask):
        raise ValueError("Fiber window is empty")
    fiber_distance_m = distance_axis_m[fiber_mask][:: args.sample_stride]

    parity_payload = {}
    all_view_values = []
    shared_reset_times_s = None
    shared_dominant_period_s = None
    if args.shared_reset_detection:
        shared_global_indices = np.arange(data.shape[0], dtype=np.int64)
        shared_data = data[:, fiber_mask][:, :: args.sample_stride]
        if args.reset_detection_end_time_s is not None:
            keep = shared_global_indices.astype(np.float64) / float(result["scan_rate"]) <= float(args.reset_detection_end_time_s)
            shared_global_indices = shared_global_indices[keep]
            shared_data = shared_data[keep]
        shared_reset_times_s, shared_dominant_period_s = detect_reset_times(
            shared_data,
            shared_global_indices,
            result["scan_rate"],
            rolling_window=args.rolling_window,
            min_period_s=args.min_period_s,
            prominence_sigma=args.prominence_sigma,
            refine_window_fraction=args.refine_window_fraction,
            max_period_s=args.max_period_s,
        )
        shared_reset_times_s = shared_reset_times_s + 1e-3 * float(args.reset_time_shift_ms)
        if args.max_reset_time_s is not None:
            shared_reset_times_s = shared_reset_times_s[shared_reset_times_s <= float(args.max_reset_time_s)]
        if args.regularize_reset_grid:
            shared_reset_times_s = regularize_reset_grid(shared_reset_times_s, shared_dominant_period_s)
        if args.reset_period_override_ms is not None:
            override_period_s = 1e-3 * float(args.reset_period_override_ms)
            if args.reset_anchor_time_s is None:
                if shared_reset_times_s.size == 0:
                    raise ValueError("Cannot infer reset anchor from an empty reset list")
                anchor_time_s = float(shared_reset_times_s[0])
            else:
                anchor_time_s = float(args.reset_anchor_time_s)
            max_grid_time_s = float(args.max_reset_time_s) if args.max_reset_time_s is not None else float(shared_global_indices[-1]) / float(result["scan_rate"])
            shared_reset_times_s = build_periodic_reset_grid(
                anchor_time_s=anchor_time_s,
                period_s=override_period_s,
                min_time_s=0.0,
                max_time_s=max_grid_time_s,
            )
            shared_dominant_period_s = override_period_s

    for parity in ["even", "odd"]:
        parity_data, parity_global_indices = select_parity_subset(data[:, fiber_mask], parity)
        parity_data = parity_data[:, :: args.sample_stride]
        parity_time_s = parity_global_indices.astype(np.float64) / float(result["scan_rate"])
        if args.shared_reset_detection:
            reset_times_s = shared_reset_times_s.copy()
            dominant_period_s = float(shared_dominant_period_s)
        else:
            detection_indices = parity_global_indices
            detection_data = parity_data
            if args.reset_detection_end_time_s is not None:
                keep = detection_indices.astype(np.float64) / float(result["scan_rate"]) <= float(args.reset_detection_end_time_s)
                detection_indices = detection_indices[keep]
                detection_data = detection_data[keep]
            reset_times_s, dominant_period_s = detect_reset_times(
                detection_data,
                detection_indices,
                result["scan_rate"],
                rolling_window=args.rolling_window,
                min_period_s=args.min_period_s,
                prominence_sigma=args.prominence_sigma,
                refine_window_fraction=args.refine_window_fraction,
                max_period_s=args.max_period_s,
            )
            reset_times_s = reset_times_s + 1e-3 * float(args.reset_time_shift_ms)
            if args.max_reset_time_s is not None:
                reset_times_s = reset_times_s[reset_times_s <= float(args.max_reset_time_s)]
            if args.regularize_reset_grid:
                reset_times_s = regularize_reset_grid(reset_times_s, dominant_period_s)
            if args.reset_period_override_ms is not None:
                override_period_s = 1e-3 * float(args.reset_period_override_ms)
                if args.reset_anchor_time_s is None:
                    if reset_times_s.size == 0:
                        raise ValueError("Cannot infer reset anchor from an empty reset list")
                    anchor_time_s = float(reset_times_s[0])
                else:
                    anchor_time_s = float(args.reset_anchor_time_s)
                max_grid_time_s = float(args.max_reset_time_s) if args.max_reset_time_s is not None else float(detection_indices[-1]) / float(result["scan_rate"])
                reset_times_s = build_periodic_reset_grid(
                    anchor_time_s=anchor_time_s,
                    period_s=override_period_s,
                    min_time_s=0.0,
                    max_time_s=max_grid_time_s,
                )
                dominant_period_s = override_period_s
        parity_payload[parity] = {
            "data": parity_data,
            "time_s": parity_time_s,
            "reset_times_s": reset_times_s,
            "dominant_period_s": dominant_period_s,
        }
        all_view_values.append(parity_data[:: args.trace_stride_full])

    stacked_for_limits = np.concatenate([arr.ravel() for arr in all_view_values])
    vmin = float(np.percentile(stacked_for_limits, args.lower_percentile))
    vmax = float(np.percentile(stacked_for_limits, args.upper_percentile))

    full_fig, full_axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for ax, parity in zip(full_axes, ["even", "odd"]):
        payload = parity_payload[parity]
        view = payload["data"][:: args.trace_stride_full]
        time_s = payload["time_s"][:: args.trace_stride_full]
        im = render_colormap(
            ax,
            view,
            fiber_distance_m,
            time_s,
            payload["reset_times_s"],
            f"{dat_path.name}: {parity} traces, baseline-subtracted",
            vmin,
            vmax,
        )
        full_fig.colorbar(im, ax=ax, label="Сигнал")
    suffix = (
        f"fiber_{int(round(args.fiber_z_min))}_{int(round(args.fiber_z_max))}m"
        f"_even_odd_colormap_resets_baseline_tail_{int(round(args.baseline_tail_m))}m"
    )
    full_png_path = output_dir / f"{dat_path.stem}_{suffix}.png"
    full_fig.savefig(full_png_path, dpi=200)
    plt.close(full_fig)

    zoom_fig, zoom_axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for ax, parity in zip(zoom_axes, ["even", "odd"]):
        payload = parity_payload[parity]
        view = payload["data"][:: args.trace_stride_zoom]
        time_s = payload["time_s"][:: args.trace_stride_zoom]
        im = render_colormap(
            ax,
            view,
            fiber_distance_m,
            time_s,
            payload["reset_times_s"],
            f"{dat_path.name}: {parity} traces, zoomed",
            vmin,
            vmax,
        )
        y0, y1 = zoom_limits(
            payload["reset_times_s"],
            payload["dominant_period_s"],
            float(time_s[-1]),
        )
        ax.set_ylim(y0, y1)
        zoom_fig.colorbar(im, ax=ax, label="Сигнал")
    zoom_png_path = output_dir / f"{dat_path.stem}_{suffix}_zoom.png"
    zoom_fig.savefig(zoom_png_path, dpi=200)
    plt.close(zoom_fig)

    tail_fig, tail_axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    tail_zoom_start_s = None
    tail_zoom_end_s = None
    for ax, parity in zip(tail_axes, ["even", "odd"]):
        payload = parity_payload[parity]
        view = payload["data"][:: args.trace_stride_zoom]
        time_s = payload["time_s"][:: args.trace_stride_zoom]
        im = render_colormap(
            ax,
            view,
            fiber_distance_m,
            time_s,
            payload["reset_times_s"],
            f"{dat_path.name}: {parity} traces, last modulation periods",
            vmin,
            vmax,
        )
        y0, y1 = tail_zoom_limits(
            payload["reset_times_s"],
            payload["dominant_period_s"],
            float(time_s[-1]),
            args.tail_period_count,
        )
        if tail_zoom_start_s is None:
            tail_zoom_start_s = y0
            tail_zoom_end_s = y1
        ax.set_ylim(y0, y1)
        tail_fig.colorbar(im, ax=ax, label="Сигнал")
    tail_zoom_png_path = output_dir / f"{dat_path.stem}_{suffix}_tail_zoom.png"
    tail_fig.savefig(tail_zoom_png_path, dpi=200)
    plt.close(tail_fig)

    matlab_payload = {
        "fiber_distance_m": fiber_distance_m,
        "even_data": parity_payload["even"]["data"][:: args.trace_stride_zoom],
        "odd_data": parity_payload["odd"]["data"][:: args.trace_stride_zoom],
        "even_time_s": parity_payload["even"]["time_s"][:: args.trace_stride_zoom],
        "odd_time_s": parity_payload["odd"]["time_s"][:: args.trace_stride_zoom],
        "reset_times_s": parity_payload["even"]["reset_times_s"],
        "tail_zoom_start_s": float(tail_zoom_start_s),
        "tail_zoom_end_s": float(tail_zoom_end_s),
        "dominant_period_s": float(parity_payload["even"]["dominant_period_s"]),
    }
    matlab_suffix = f"{suffix}_tail_zoom"
    matlab_mat_path, matlab_script_path = save_matlab_bundle(output_dir, dat_path.stem, matlab_suffix, matlab_payload)

    print(f"file: {dat_path}")
    print(f"shape: {result['data'].shape}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"distance_step_m: {distance_step_m:.10f}")
    print(f"fiber_distance_start_m: {fiber_distance_m[0]:.6f}")
    print(f"fiber_distance_end_m: {fiber_distance_m[-1]:.6f}")
    print(f"baseline_tail_m: {args.baseline_tail_m}")
    print(f"baseline_window_start_m: {baseline_start_m:.6f}")
    print(f"baseline_window_end_m: {baseline_end_m:.6f}")
    print(f"baseline_mean_over_traces: {np.mean(baseline_per_trace):.10e}")
    print(f"baseline_std_over_traces: {np.std(baseline_per_trace):.10e}")
    print(f"reset_time_shift_ms: {args.reset_time_shift_ms}")
    print(f"shared_reset_detection: {args.shared_reset_detection}")
    print(f"regularize_reset_grid: {args.regularize_reset_grid}")
    print(f"tail_period_count: {args.tail_period_count}")
    if args.max_period_s is not None:
        print(f"max_period_s: {args.max_period_s}")
    if args.max_reset_time_s is not None:
        print(f"max_reset_time_s: {args.max_reset_time_s}")
    if args.reset_detection_end_time_s is not None:
        print(f"reset_detection_end_time_s: {args.reset_detection_end_time_s}")
    if args.shared_reset_detection:
        print(f"shared_dominant_period_s: {shared_dominant_period_s:.10f}")
        print(f"shared_reset_count: {shared_reset_times_s.size}")
    for parity in ["even", "odd"]:
        print(f"{parity}_dominant_period_s: {parity_payload[parity]['dominant_period_s']:.10f}")
        print(f"{parity}_reset_count: {parity_payload[parity]['reset_times_s'].size}")
    print(f"full_colormap_png_saved_to: {full_png_path}")
    print(f"zoom_colormap_png_saved_to: {zoom_png_path}")
    print(f"tail_zoom_colormap_png_saved_to: {tail_zoom_png_path}")
    print(f"matlab_data_saved_to: {matlab_mat_path}")
    print(f"matlab_script_saved_to: {matlab_script_path}")


if __name__ == "__main__":
    main()
