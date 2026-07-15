import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from detect_period_resets import (
    detect_period_locked_spikes,
    dominant_score,
    estimate_period_from_fft,
    frame_change_metric,
    moving_average,
    refine_peaks_with_metric,
    robust_threshold,
    select_parity_subset,
)
from raw_data import read_reflectograms
from reflectometer_utils import distance_axis_from_sampling_rate, subtract_trace_baseline_from_tail


def matlab_safe_stem(text):
    safe = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not safe or not safe[0].isalpha():
        safe = f"fig_{safe}"
    return safe


def detect_reset_times(
    parity_data,
    parity_global_indices,
    scan_rate_hz,
    rolling_window,
    min_period_s,
    prominence_sigma,
    refine_window_fraction,
    max_period_s=None,
):
    score = dominant_score(parity_data)
    if np.mean(np.diff(score[: min(5000, score.size - 1)])) < 0:
        score = -score

    score_smooth = moving_average(score, rolling_window)
    global_step = float(np.median(np.diff(parity_global_indices))) if parity_global_indices.size > 1 else 1.0
    effective_scan_rate_hz = float(scan_rate_hz) / global_step
    duration_s = (parity_global_indices[-1] - parity_global_indices[0]) / float(scan_rate_hz)
    dominant_period_s, _, _, _, _ = estimate_period_from_fft(
        score_smooth,
        effective_scan_rate_hz,
        min_period_s=min_period_s,
        max_period_s=(0.5 * duration_s) if max_period_s is None else float(max_period_s),
    )
    period_traces = max(2, int(round(dominant_period_s * effective_scan_rate_hz)))
    phase_half_width_traces = max(1, int(round(0.3 * period_traces)))
    change_metric = frame_change_metric(parity_data)
    amplitude_threshold = robust_threshold(change_metric, prominence_sigma)
    coarse_peaks, _, _, _ = detect_period_locked_spikes(
        change_metric,
        period_traces=period_traces,
        amplitude_threshold=amplitude_threshold,
        phase_half_width_traces=phase_half_width_traces,
    )
    refine_window_traces = max(1, int(round(float(refine_window_fraction) * period_traces)))
    refined_peaks, _ = refine_peaks_with_metric(
        coarse_peaks,
        change_metric,
        search_radius_traces=refine_window_traces,
    )
    derivative_time_s = (parity_global_indices[:-1] + parity_global_indices[1:]) / (2.0 * float(scan_rate_hz))
    reset_times_s = derivative_time_s[refined_peaks] if refined_peaks.size > 0 else np.array([], dtype=np.float64)
    return reset_times_s, dominant_period_s


def build_sweep_intervals(reset_times_s):
    reset_times_s = np.asarray(reset_times_s, dtype=np.float64)
    if reset_times_s.size < 2:
        raise ValueError("Need at least two reset times to form sweep intervals")
    return np.column_stack([reset_times_s[:-1], reset_times_s[1:]])


def harmonics_for_sweeps(parity_data, parity_time_s, sweep_intervals_s, delta_beta_span, lag_distances_m):
    harmonic_cube = []
    h0_rows = []
    sweep_rows = []
    for sweep_index, (t0, t1) in enumerate(sweep_intervals_s):
        sample_mask = (parity_time_s >= t0) & (parity_time_s < t1)
        if np.count_nonzero(sample_mask) < (2 * lag_distances_m.size + 4):
            continue

        sweep_signal_raw = parity_data[sample_mask]
        h0_vector = np.mean(sweep_signal_raw, axis=0)
        sweep_signal = sweep_signal_raw - h0_vector[None, :]
        local_time = parity_time_s[sample_mask]
        phase_fraction = (local_time - t0) / (t1 - t0)
        delta_beta = phase_fraction * delta_beta_span

        harmonics = np.empty((sweep_signal.shape[1], lag_distances_m.size), dtype=np.complex128)
        for lag_idx, lag_distance_m in enumerate(lag_distances_m):
            basis = np.exp(-1j * 2.0 * delta_beta * float(lag_distance_m))
            harmonics[:, lag_idx] = np.mean(sweep_signal * basis[:, None], axis=0)
        harmonic_cube.append(harmonics)
        h0_rows.append(h0_vector)
        sweep_rows.append(
            {
                "sweep_index": int(sweep_index),
                "start_s": float(t0),
                "end_s": float(t1),
                "sample_count": int(np.count_nonzero(sample_mask)),
                "mean_h0": np.mean(h0_vector),
                "mean_abs_harmonics": np.mean(np.abs(harmonics), axis=0),
            }
        )

    if len(harmonic_cube) == 0:
        raise ValueError("No complete sweep intervals survived harmonic fitting")

    return np.stack(harmonic_cube, axis=0), np.stack(h0_rows, axis=0), sweep_rows


def write_mean_abs_csv(path, lag_indices, sweep_rows):
    lag_indices = np.asarray(lag_indices, dtype=np.int64)
    with Path(path).open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        header = ["sweep_index", "start_s", "end_s", "sample_count", "H0_mean"] + [f"H{int(p)}_mean_abs" for p in lag_indices]
        writer.writerow(header)
        for row in sweep_rows:
            writer.writerow(
                [
                    row["sweep_index"],
                    f"{row['start_s']:.10f}",
                    f"{row['end_s']:.10f}",
                    row["sample_count"],
                    f"{row['mean_h0']:.10e}",
                    *[f"{value:.10e}" for value in row["mean_abs_harmonics"]],
                ]
            )


def write_coordinate_mean_abs_csv(path, fiber_distance_m, lag_indices, coordinate_mean_abs):
    fiber_distance_m = np.asarray(fiber_distance_m, dtype=np.float64)
    lag_indices = np.asarray(lag_indices, dtype=np.int64)
    coordinate_mean_abs = np.asarray(coordinate_mean_abs, dtype=np.float64)
    with Path(path).open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        header = ["coordinate_index", "distance_m"] + [f"H{int(p)}_mean_abs" for p in lag_indices]
        writer.writerow(header)
        for coord_index, (distance_m, row) in enumerate(zip(fiber_distance_m, coordinate_mean_abs)):
            writer.writerow(
                [
                    coord_index,
                    f"{distance_m:.10f}",
                    *[f"{value:.10e}" for value in row],
                ]
            )


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    data_mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"

    savemat(data_mat_path, payload)

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{data_mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Mean abs harmonics vs sweep');
subplot(2,1,1);
imagesc(data.lag_indices, data.even_sweep_index, data.even_mean_abs_harmonics);
axis xy; colorbar; xlabel('Lag p'); ylabel('Even sweep index'); title('Even mean |H_p|');
subplot(2,1,2);
imagesc(data.lag_indices, data.odd_sweep_index, data.odd_mean_abs_harmonics);
axis xy; colorbar; xlabel('Lag p'); ylabel('Odd sweep index'); title('Odd mean |H_p|');

f2 = figure('Color', 'w', 'Name', 'Sweep-averaged harmonics vs lag');
plot(data.lag_indices, data.even_mean_abs_over_sweeps, 'LineWidth', 1.6);
hold on;
plot(data.lag_indices, data.odd_mean_abs_over_sweeps, 'LineWidth', 1.6);
grid on; xlabel('Lag p'); ylabel('Mean |H_p|');
title('Sweep-averaged mean absolute harmonics');
legend('Even', 'Odd', 'Location', 'best');

f3 = figure('Color', 'w', 'Name', 'Coordinate-resolved harmonics');
subplot(2,1,1);
imagesc(data.lag_indices, data.fiber_distance_m, data.even_coordinate_mean_abs_harmonics);
axis xy; colorbar; xlabel('Lag p'); ylabel('Distance (m)'); title('Even mean |H_p(z)| over sweeps');
subplot(2,1,2);
imagesc(data.lag_indices, data.fiber_distance_m, data.odd_coordinate_mean_abs_harmonics);
axis xy; colorbar; xlabel('Lag p'); ylabel('Distance (m)'); title('Odd mean |H_p(z)| over sweeps');

f4 = figure('Color', 'w', 'Name', 'H0 vs coordinate');
plot(data.fiber_distance_m, data.even_mean_h0_over_sweeps, 'LineWidth', 1.6);
hold on;
plot(data.fiber_distance_m, data.odd_mean_h0_over_sweeps, 'LineWidth', 1.6);
grid on; xlabel('Distance (m)'); ylabel('H0');
title('Sweep-averaged H0(z)');
legend('Even', 'Odd', 'Location', 'best');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return {"mat": data_mat_path, "script": script_path}


def main():
    parser = argparse.ArgumentParser(
        description="Рассчитать гармоники по лагам для каждого свипа длины волны из чётных и нечётных рефлектограмм."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Каталог для выходных файлов")
    parser.add_argument("--scan-rate", type=float, default=None, help="Необязательная частота записи рефлектограмм в Hz")
    parser.add_argument("--fiber-z-min", type=float, default=100.0, help="Начало полезного участка волокна в метрах")
    parser.add_argument("--fiber-z-max", type=float, default=280.0, help="Конец полезного участка волокна в метрах")
    parser.add_argument("--pulse-z-min", type=float, default=75.0, help="Начало поддержки импульса в метрах")
    parser.add_argument("--pulse-z-max", type=float, default=85.0, help="Конец поддержки импульса в метрах")
    parser.add_argument("--lambda0-nm", type=float, default=1550.0, help="Центральная длина волны в nm")
    parser.add_argument("--sweep-span-pm", type=float, default=0.78, help="Размах одного свипа длины волны в pm")
    parser.add_argument("--rolling-window", type=int, default=64, help="Окно сглаживания детектора сбросов в трассах")
    parser.add_argument("--min-period-s", type=float, default=0.05, help="Минимальный период свипа для детектора сбросов")
    parser.add_argument("--prominence-sigma", type=float, default=3.0, help="Порог детектора сбросов в робастных sigma")
    parser.add_argument("--refine-window-fraction", type=float, default=0.15, help="Окно локального уточнения как доля найденного периода")
    parser.add_argument("--reset-time-shift-ms", type=float, default=0.0, help="Сдвинуть найденные времена сбросов позже на это число ms")
    parser.add_argument("--plot-sweep-index", type=int, default=0, help="Индекс свипа с нуля для построения карты |H_n(z)|")
    parser.add_argument("--baseline-tail-m", type=float, default=None, help="Вычесть baseline каждой трассы, оцененный по последним N метрам")
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float64)
    distance_axis_m, distance_step_m, n_eff = distance_axis_from_sampling_rate(
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
    fiber_mask = (distance_axis_m >= float(args.fiber_z_min)) & (distance_axis_m <= float(args.fiber_z_max))
    if not np.any(pulse_mask):
        raise ValueError("Pulse support window is empty")
    if not np.any(fiber_mask):
        raise ValueError("Fiber window is empty")

    pulse_distance_m = distance_axis_m[pulse_mask]
    fiber_distance_m = distance_axis_m[fiber_mask]
    pulse_discrete_count = int(pulse_distance_m.size)
    if pulse_discrete_count < 2:
        raise ValueError("Need at least two pulse discretes")

    lag_indices = np.arange(1, pulse_discrete_count, dtype=np.int64)
    lag_distances_m = lag_indices.astype(np.float64) * float(distance_step_m)
    lambda0_m = float(args.lambda0_nm) * 1e-9
    sweep_span_m = float(args.sweep_span_pm) * 1e-12
    delta_beta_span = -2.0 * np.pi * float(n_eff) * sweep_span_m / (lambda0_m**2)

    parity_payload = {}
    for parity in ["even", "odd"]:
        fiber_data, parity_global_indices = select_parity_subset(data[:, fiber_mask], parity)
        parity_time_s = parity_global_indices.astype(np.float64) / float(result["scan_rate"])
        reset_times_s, dominant_period_s = detect_reset_times(
            fiber_data,
            parity_global_indices,
            result["scan_rate"],
            rolling_window=args.rolling_window,
            min_period_s=args.min_period_s,
            prominence_sigma=args.prominence_sigma,
            refine_window_fraction=args.refine_window_fraction,
        )
        reset_times_s = reset_times_s + 1e-3 * float(args.reset_time_shift_ms)
        sweep_intervals_s = build_sweep_intervals(reset_times_s)
        harmonic_cube, h0_matrix, sweep_rows = harmonics_for_sweeps(
            fiber_data,
            parity_time_s,
            sweep_intervals_s,
            delta_beta_span=delta_beta_span,
            lag_distances_m=lag_distances_m,
        )
        mean_abs_harmonics = np.stack([row["mean_abs_harmonics"] for row in sweep_rows], axis=0)
        coordinate_mean_abs_harmonics = np.mean(np.abs(harmonic_cube), axis=0)
        mean_h0_over_sweeps = np.mean(h0_matrix, axis=0)
        std_h0_over_sweeps = np.std(h0_matrix, axis=0)
        parity_payload[parity] = {
            "harmonics": harmonic_cube,
            "h0_matrix": h0_matrix,
            "sweep_rows": sweep_rows,
            "mean_abs_harmonics": mean_abs_harmonics,
            "coordinate_mean_abs_harmonics": coordinate_mean_abs_harmonics,
            "mean_h0_over_sweeps": mean_h0_over_sweeps,
            "std_h0_over_sweeps": std_h0_over_sweeps,
            "dominant_period_s": dominant_period_s,
            "reset_times_s": reset_times_s,
        }

        csv_path = output_dir / f"{dat_path.stem}_harmonics_{parity}_mean_abs.csv"
        write_mean_abs_csv(csv_path, lag_indices, sweep_rows)
        coord_csv_path = output_dir / f"{dat_path.stem}_harmonics_{parity}_coordinate_mean_abs.csv"
        write_coordinate_mean_abs_csv(coord_csv_path, fiber_distance_m, lag_indices, coordinate_mean_abs_harmonics)

    heatmap_fig, heatmap_axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, parity in zip(heatmap_axes, ["even", "odd"]):
        matrix = parity_payload[parity]["mean_abs_harmonics"]
        sweep_index = np.arange(matrix.shape[0], dtype=np.int64)
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[lag_indices[0], lag_indices[-1], sweep_index[0], sweep_index[-1]],
        )
        ax.set_ylabel(f"{parity} свип")
        ax.set_title(f"{parity.capitalize()} среднее |H_p| по свипам длины волны")
        heatmap_fig.colorbar(im, ax=ax, label="Среднее |H_p| по волокну")
    heatmap_axes[-1].set_xlabel("Лаг p")
    heatmap_png_path = output_dir / f"{dat_path.stem}_harmonics_mean_abs_heatmap.png"
    heatmap_fig.savefig(heatmap_png_path, dpi=200)
    plt.close(heatmap_fig)

    line_fig, line_ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for parity, color in [("even", "#4C78A8"), ("odd", "#F58518")]:
        mean_abs_over_sweeps = np.mean(parity_payload[parity]["mean_abs_harmonics"], axis=0)
        line_ax.plot(lag_indices, mean_abs_over_sweeps, linewidth=1.8, label=f"{parity} mean |H_p|", color=color)
    line_ax.set_xlabel("Лаг p")
    line_ax.set_ylabel("Среднее |H_p|")
    line_ax.set_title("Средние по свипам абсолютные гармоники")
    line_ax.grid(alpha=0.25)
    line_ax.legend()
    line_png_path = output_dir / f"{dat_path.stem}_harmonics_mean_abs_vs_lag.png"
    line_fig.savefig(line_png_path, dpi=200)
    plt.close(line_fig)

    coord_fig, coord_axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, parity in zip(coord_axes, ["even", "odd"]):
        matrix = parity_payload[parity]["coordinate_mean_abs_harmonics"]
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[lag_indices[0], lag_indices[-1], fiber_distance_m[0], fiber_distance_m[-1]],
        )
        ax.set_ylabel("Расстояние (m)")
        ax.set_title(f"{parity.capitalize()} среднее |H_p(z)| по свипам")
        coord_fig.colorbar(im, ax=ax, label="Среднее |H_p(z)|")
    coord_axes[-1].set_xlabel("Лаг p")
    coord_png_path = output_dir / f"{dat_path.stem}_harmonics_coordinate_mean_abs_heatmap.png"
    coord_fig.savefig(coord_png_path, dpi=200)
    plt.close(coord_fig)

    h0_fig, h0_ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for parity, color in [("even", "#4C78A8"), ("odd", "#F58518")]:
        mean_h0 = parity_payload[parity]["mean_h0_over_sweeps"]
        std_h0 = parity_payload[parity]["std_h0_over_sweeps"]
        h0_ax.plot(fiber_distance_m, mean_h0, linewidth=1.6, color=color, label=f"{parity} mean H0")
        h0_ax.fill_between(fiber_distance_m, mean_h0 - std_h0, mean_h0 + std_h0, color=color, alpha=0.18)
    h0_ax.set_xlabel("Расстояние (m)")
    h0_ax.set_ylabel("H0")
    h0_ax.set_title("H0(z), усреднённый по свипам")
    h0_ax.grid(alpha=0.25)
    h0_ax.legend()
    h0_png_path = output_dir / f"{dat_path.stem}_harmonics_H0_vs_coordinate.png"
    h0_fig.savefig(h0_png_path, dpi=200)
    plt.close(h0_fig)

    available_sweeps = min(
        parity_payload["even"]["harmonics"].shape[0],
        parity_payload["odd"]["harmonics"].shape[0],
    )
    if args.plot_sweep_index < 0 or args.plot_sweep_index >= available_sweeps:
        raise ValueError(
            f"plot_sweep_index must be between 0 and {available_sweeps - 1}, got {args.plot_sweep_index}"
        )

    single_fig, single_axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, parity in zip(single_axes, ["even", "odd"]):
        matrix = np.abs(parity_payload[parity]["harmonics"][args.plot_sweep_index])
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[lag_indices[0], lag_indices[-1], fiber_distance_m[0], fiber_distance_m[-1]],
        )
        ax.set_ylabel("Расстояние (m)")
        ax.set_title(f"{parity.capitalize()} |H_n(z)| для свипа {args.plot_sweep_index}")
        single_fig.colorbar(im, ax=ax, label="|H_n(z)|")
    single_axes[-1].set_xlabel("Лаг n")
    single_sweep_png_path = output_dir / f"{dat_path.stem}_harmonics_sweep_{args.plot_sweep_index}_coordinate_heatmap.png"
    single_fig.savefig(single_sweep_png_path, dpi=200)
    plt.close(single_fig)

    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag="sweep_harmonics_even_odd",
        payload={
            "lag_indices": lag_indices[:, None],
            "lag_distances_m": lag_distances_m[:, None],
            "fiber_distance_m": fiber_distance_m[:, None],
            "pulse_distance_m": pulse_distance_m[:, None],
            "distance_step_m": np.array([[distance_step_m]], dtype=np.float64),
            "pulse_discrete_count": np.array([[pulse_discrete_count]], dtype=np.int32),
            "sweep_span_pm": np.array([[args.sweep_span_pm]], dtype=np.float64),
            "delta_beta_span": np.array([[delta_beta_span]], dtype=np.float64),
            "even_harmonics": parity_payload["even"]["harmonics"],
            "odd_harmonics": parity_payload["odd"]["harmonics"],
            "even_h0_matrix": parity_payload["even"]["h0_matrix"],
            "odd_h0_matrix": parity_payload["odd"]["h0_matrix"],
            "even_mean_abs_harmonics": parity_payload["even"]["mean_abs_harmonics"],
            "odd_mean_abs_harmonics": parity_payload["odd"]["mean_abs_harmonics"],
            "even_coordinate_mean_abs_harmonics": parity_payload["even"]["coordinate_mean_abs_harmonics"],
            "odd_coordinate_mean_abs_harmonics": parity_payload["odd"]["coordinate_mean_abs_harmonics"],
            "even_mean_abs_over_sweeps": np.mean(parity_payload["even"]["mean_abs_harmonics"], axis=0)[None, :],
            "odd_mean_abs_over_sweeps": np.mean(parity_payload["odd"]["mean_abs_harmonics"], axis=0)[None, :],
            "even_mean_h0_over_sweeps": parity_payload["even"]["mean_h0_over_sweeps"][:, None],
            "odd_mean_h0_over_sweeps": parity_payload["odd"]["mean_h0_over_sweeps"][:, None],
            "even_sweep_index": np.arange(parity_payload["even"]["mean_abs_harmonics"].shape[0], dtype=np.int32)[:, None],
            "odd_sweep_index": np.arange(parity_payload["odd"]["mean_abs_harmonics"].shape[0], dtype=np.int32)[:, None],
            "even_reset_times_s": parity_payload["even"]["reset_times_s"][:, None],
            "odd_reset_times_s": parity_payload["odd"]["reset_times_s"][:, None],
        },
    )

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"distance_step_m: {distance_step_m:.10f}")
    print(f"pulse_discrete_count_N: {pulse_discrete_count}")
    print(f"pulse_distance_start_m: {pulse_distance_m[0]:.6f}")
    print(f"pulse_distance_end_m: {pulse_distance_m[-1]:.6f}")
    print(f"fiber_distance_start_m: {fiber_distance_m[0]:.6f}")
    print(f"fiber_distance_end_m: {fiber_distance_m[-1]:.6f}")
    if args.baseline_tail_m is not None:
        print(f"baseline_tail_m: {args.baseline_tail_m}")
        print(f"baseline_window_start_m: {baseline_start_m:.6f}")
        print(f"baseline_window_end_m: {baseline_end_m:.6f}")
    print(f"reset_time_shift_ms: {args.reset_time_shift_ms}")
    print(f"sweep_span_pm: {args.sweep_span_pm}")
    print(f"delta_beta_span: {delta_beta_span:.10e}")
    for parity in ["even", "odd"]:
        print(f"{parity}_dominant_period_s: {parity_payload[parity]['dominant_period_s']:.10f}")
        print(f"{parity}_reset_count: {parity_payload[parity]['reset_times_s'].size}")
        print(f"{parity}_complete_sweeps: {parity_payload[parity]['mean_abs_harmonics'].shape[0]}")
        print(
            f"{parity}_mean_abs_H1_over_sweeps: "
            f"{np.mean(parity_payload[parity]['mean_abs_harmonics'][:, 0]):.10e}"
        )
        print(
            f"{parity}_mean_abs_last_harmonic_over_sweeps: "
            f"{np.mean(parity_payload[parity]['mean_abs_harmonics'][:, -1]):.10e}"
        )
        print(
            f"{parity}_mean_H0_over_coordinates: "
            f"{np.mean(parity_payload[parity]['mean_h0_over_sweeps']):.10e}"
        )
        print(
            f"{parity}_std_H0_over_coordinates: "
            f"{np.std(parity_payload[parity]['mean_h0_over_sweeps']):.10e}"
        )
        print(f"{parity}_csv_saved_to: {output_dir / f'{dat_path.stem}_harmonics_{parity}_mean_abs.csv'}")
        print(
            f"{parity}_coordinate_csv_saved_to: "
            f"{output_dir / f'{dat_path.stem}_harmonics_{parity}_coordinate_mean_abs.csv'}"
        )
    print(f"heatmap_png_saved_to: {heatmap_png_path}")
    print(f"line_png_saved_to: {line_png_path}")
    print(f"coordinate_heatmap_png_saved_to: {coord_png_path}")
    print(f"h0_png_saved_to: {h0_png_path}")
    print(f"single_sweep_index: {args.plot_sweep_index}")
    print(f"single_sweep_heatmap_png_saved_to: {single_sweep_png_path}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
