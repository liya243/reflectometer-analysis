import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from analysis_output_utils import cleanup_outputs_for_dataset
from detect_period_resets import frame_change_metric, moving_average, robust_threshold
from raw_data import read_reflectograms
from reflectometer_utils import distance_axis_from_sampling_rate, subtract_trace_baseline_from_tail


def find_local_peaks(values, threshold, min_distance_samples):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return np.array([], dtype=np.int64)

    candidate_mask = (
        (values[1:-1] >= values[:-2])
        & (values[1:-1] > values[2:])
        & (values[1:-1] >= float(threshold))
    )
    candidate_indices = np.flatnonzero(candidate_mask) + 1
    if candidate_indices.size == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(values[candidate_indices])[::-1]
    chosen = []
    blocked = np.zeros(values.size, dtype=bool)
    for idx in candidate_indices[order]:
        left = max(0, int(idx) - int(min_distance_samples))
        right = min(values.size, int(idx) + int(min_distance_samples) + 1)
        if np.any(blocked[left:right]):
            continue
        chosen.append(int(idx))
        blocked[left:right] = True

    if not chosen:
        return np.array([], dtype=np.int64)
    return np.asarray(sorted(chosen), dtype=np.int64)


def render_colormap(ax, view, distance_m, time_s, jump_times_s, title, vmin, vmax):
    im = ax.imshow(
        view,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[distance_m[0], distance_m[-1], time_s[0], time_s[-1]],
    )
    for jump_time_s in np.asarray(jump_times_s, dtype=np.float64):
        ax.axhline(float(jump_time_s), color="#D62728", linewidth=0.8, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")
    return im


def active_zoom_limits(jump_times_s, record_end_s, margin_s, fallback_duration_s):
    jump_times_s = np.asarray(jump_times_s, dtype=np.float64)
    if jump_times_s.size == 0:
        return 0.0, min(float(record_end_s), float(fallback_duration_s))

    start_s = max(0.0, float(jump_times_s[0]) - float(margin_s))
    end_s = min(float(record_end_s), float(jump_times_s[-1]) + float(margin_s))
    if end_s <= start_s:
        end_s = min(float(record_end_s), start_s + float(fallback_duration_s))
    return start_s, end_s


def main():
    parser = argparse.ArgumentParser(
        description="Render a baseline-subtracted fiber colormap and mark abrupt modulation jumps."
    )
    parser.add_argument("dat_path", help="Path to the .dat file")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Directory for output files")
    parser.add_argument("--scan-rate", type=float, default=None, help="Optional override for reflectogram scan rate in Hz")
    parser.add_argument("--fiber-z-min", type=float, default=100.0, help="Start of useful reflectogram region in meters")
    parser.add_argument("--fiber-z-max", type=float, default=200.0, help="End of useful reflectogram region in meters")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Subtract per-trace baseline from the last this many meters")
    parser.add_argument("--trace-stride-full", type=int, default=8, help="Keep every Nth trace in the full-time colormap")
    parser.add_argument("--trace-stride-zoom", type=int, default=1, help="Keep every Nth trace in the zoomed colormap")
    parser.add_argument("--sample-stride", type=int, default=1, help="Keep every Nth sample along distance")
    parser.add_argument("--detection-smooth-window", type=int, default=5, help="Smoothing window for frame-to-frame jump score")
    parser.add_argument("--jump-threshold-sigma", type=float, default=5.0, help="Jump detector threshold in robust sigma units")
    parser.add_argument("--min-peak-distance-s", type=float, default=0.03, help="Minimum time separation between detected jumps")
    parser.add_argument("--zoom-margin-s", type=float, default=0.12, help="Zoom margin before first and after last detected jump")
    parser.add_argument("--zoom-fallback-duration-s", type=float, default=1.0, help="Zoom duration if no jumps are found")
    parser.add_argument("--lower-percentile", type=float, default=1.0, help="Lower percentile for color clipping")
    parser.add_argument("--upper-percentile", type=float, default=99.0, help="Upper percentile for color clipping")
    parser.add_argument("--cleanup-dataset-outputs", action="store_true", help="Delete previous outputs for this dataset before saving new ones")
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.cleanup_dataset_outputs:
        cleanup_outputs_for_dataset(output_dir, dat_path.stem)

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
        raise ValueError("Useful reflectogram window is empty")

    fiber_distance_m = distance_axis_m[fiber_mask][:: args.sample_stride]
    fiber_data = data[:, fiber_mask][:, :: args.sample_stride]
    trace_time_s = np.arange(fiber_data.shape[0], dtype=np.float64) / float(result["scan_rate"])

    jump_metric = frame_change_metric(fiber_data)
    jump_metric_smooth = moving_average(jump_metric, args.detection_smooth_window)
    jump_threshold = robust_threshold(jump_metric_smooth, args.jump_threshold_sigma)
    min_peak_distance_traces = max(1, int(round(float(args.min_peak_distance_s) * float(result["scan_rate"]))))
    jump_indices = find_local_peaks(
        jump_metric_smooth,
        threshold=jump_threshold,
        min_distance_samples=min_peak_distance_traces,
    )
    jump_time_s = (jump_indices.astype(np.float64) + 0.5) / float(result["scan_rate"])

    stacked = fiber_data[:: args.trace_stride_full].ravel()
    vmin = float(np.percentile(stacked, args.lower_percentile))
    vmax = float(np.percentile(stacked, args.upper_percentile))

    full_fig, full_ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    full_view = fiber_data[:: args.trace_stride_full]
    full_time_s = trace_time_s[:: args.trace_stride_full]
    full_im = render_colormap(
        full_ax,
        full_view,
        fiber_distance_m,
        full_time_s,
        jump_time_s,
        f"{dat_path.name}: useful reflectogram, baseline-subtracted",
        vmin,
        vmax,
    )
    full_fig.colorbar(full_im, ax=full_ax, label="Signal")
    suffix = (
        f"fiber_{int(round(args.fiber_z_min))}_{int(round(args.fiber_z_max))}m"
        f"_colormap_jumps_baseline_tail_{int(round(args.baseline_tail_m))}m"
    )
    full_png_path = output_dir / f"{dat_path.stem}_{suffix}.png"
    full_fig.savefig(full_png_path, dpi=200)
    plt.close(full_fig)

    zoom_fig, zoom_ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    zoom_view = fiber_data[:: args.trace_stride_zoom]
    zoom_time_s = trace_time_s[:: args.trace_stride_zoom]
    zoom_im = render_colormap(
        zoom_ax,
        zoom_view,
        fiber_distance_m,
        zoom_time_s,
        jump_time_s,
        f"{dat_path.name}: useful reflectogram, zoomed modulation jumps",
        vmin,
        vmax,
    )
    y0, y1 = active_zoom_limits(jump_time_s, float(trace_time_s[-1]), args.zoom_margin_s, args.zoom_fallback_duration_s)
    zoom_ax.set_ylim(y0, y1)
    zoom_fig.colorbar(zoom_im, ax=zoom_ax, label="Signal")
    zoom_png_path = output_dir / f"{dat_path.stem}_{suffix}_zoom.png"
    zoom_fig.savefig(zoom_png_path, dpi=200)
    plt.close(zoom_fig)

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
    print(f"jump_metric_threshold: {jump_threshold:.10e}")
    print(f"jump_count: {jump_time_s.size}")
    print(f"jump_times_s: {','.join(f'{value:.6f}' for value in jump_time_s)}")
    print(f"full_colormap_png_saved_to: {full_png_path}")
    print(f"zoom_colormap_png_saved_to: {zoom_png_path}")


if __name__ == "__main__":
    main()
