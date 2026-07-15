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


def center_and_rms_normalize_rows(trace_matrix):
    data = np.asarray(trace_matrix, dtype=np.float64)
    centered = data - data.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1, keepdims=True))
    normalized = np.full_like(centered, np.nan, dtype=np.float64)
    valid = rms[:, 0] > 0.0
    normalized[valid] = centered[valid] / rms[valid]
    return normalized


def moving_average_ignore_nan(values, window):
    values = np.asarray(values, dtype=np.float64)
    if int(window) <= 1:
        return values.copy()
    kernel = np.ones(int(window), dtype=np.float64)
    filled = np.nan_to_num(values, nan=0.0)
    valid = np.isfinite(values).astype(np.float64)
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")
    out = np.full_like(values, np.nan, dtype=np.float64)
    mask = counts > 0.0
    out[mask] = sums[mask] / counts[mask]
    return out


def build_periodic_reset_grid(anchor_time_s, period_s, min_time_s, max_time_s):
    period_s = float(period_s)
    if not np.isfinite(period_s) or period_s <= 0.0:
        raise ValueError("period_s must be positive")

    first_index = int(np.ceil((float(min_time_s) - float(anchor_time_s)) / period_s))
    last_index = int(np.floor((float(max_time_s) - float(anchor_time_s)) / period_s))
    if last_index < first_index:
        return np.array([], dtype=np.float64)
    indices = np.arange(first_index, last_index + 1, dtype=np.float64)
    return float(anchor_time_s) + indices * period_s


def correlation_matrix_blocked(observed_norm, reference_norm, block_rows):
    observed_norm = np.asarray(observed_norm, dtype=np.float64)
    reference_norm = np.asarray(reference_norm, dtype=np.float64)
    out = np.empty((observed_norm.shape[0], reference_norm.shape[0]), dtype=np.float64)
    scale = float(observed_norm.shape[1])
    for start in range(0, observed_norm.shape[0], int(block_rows)):
        stop = min(observed_norm.shape[0], start + int(block_rows))
        out[start:stop] = observed_norm[start:stop] @ reference_norm.T / scale
    return out


def track_continuous_reference(correlation_matrix, reference_lambda_pm, initial_window_traces, step_sigma_pm):
    correlation_matrix = np.asarray(correlation_matrix, dtype=np.float64)
    reference_lambda_pm = np.asarray(reference_lambda_pm, dtype=np.float64).reshape(-1)
    initial_count = max(1, min(int(initial_window_traces), correlation_matrix.shape[0]))
    initial_score = np.nanmean(correlation_matrix[:initial_count], axis=0)

    tracked_index = np.empty(correlation_matrix.shape[0], dtype=np.int64)
    tracked_index[0] = int(np.nanargmax(initial_score))
    inv_two_sigma2 = 0.5 / max(float(step_sigma_pm) ** 2, 1e-12)
    for row in range(1, correlation_matrix.shape[0]):
        previous = reference_lambda_pm[tracked_index[row - 1]]
        penalty = inv_two_sigma2 * (reference_lambda_pm - previous) ** 2
        tracked_index[row] = int(np.nanargmax(correlation_matrix[row] - penalty))

    tracked_lambda_pm = reference_lambda_pm[tracked_index]
    tracked_corr = correlation_matrix[np.arange(correlation_matrix.shape[0]), tracked_index]
    return tracked_lambda_pm, tracked_corr, tracked_index


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Trace-bank wavelength coordinate');
plot(data.merged_time_abs_s, data.merged_lambda_pm, '.', 'Color', [0.65 0.65 0.65], 'MarkerSize', 4);
hold on;
plot(data.even_time_abs_s, data.even_lambda_pm, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 4);
plot(data.odd_time_abs_s, data.odd_lambda_pm, '.', 'Color', [1.00 0.50 0.05], 'MarkerSize', 4);
grid on;
xlabel('Time (s)');
ylabel('Best coordinate inside last sweep (pm)');
title('Post-sweep wavelength coordinate from trace-bank matching');
legend('Merged', 'Even', 'Odd', 'Location', 'best');

f2 = figure('Color', 'w', 'Name', 'Trace-bank drift');
plot(data.merged_time_abs_s, data.merged_drift_pm, '.', 'Color', [0.55 0.55 0.55], 'MarkerSize', 4);
hold on;
plot(data.merged_time_abs_s, data.merged_drift_pm_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Time (s)');
ylabel('Drift from initial post-sweep level (pm)');
title('Wavelength drift after laser modulation is off');

f3 = figure('Color', 'w', 'Name', 'Trace-bank fit quality');
plot(data.merged_time_abs_s, data.merged_fit_corr, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 4);
hold on;
plot(data.merged_time_abs_s, data.merged_fit_corr_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Time (s)');
ylabel('Best correlation');
title('Trace-bank matching quality');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return mat_path, script_path


def main():
    parser = argparse.ArgumentParser(
        description="Оценить дрейф длины волны после свипа сопоставлением трасс с экспериментальным банком трасс последнего свипа."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Каталог для выходных файлов")
    parser.add_argument("--scan-rate", type=float, default=None, help="Необязательная частота записи рефлектограмм")
    parser.add_argument("--fiber-z-min", type=float, default=110.0, help="Начало полезного участка волокна в метрах")
    parser.add_argument("--fiber-z-max", type=float, default=360.0, help="Конец полезного участка волокна в метрах")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Вычесть baseline каждой трассы по последним N метрам")
    parser.add_argument("--reset-period-ms", type=float, default=76.8, help="Принятый период пилообразного свипа в ms")
    parser.add_argument("--reset-anchor-time-s", type=float, default=0.0919, help="Одно принятое время сброса/границы свипа")
    parser.add_argument("--max-reset-time-s", type=float, default=4.45, help="Верхняя граница времени для принятой сетки сбросов")
    parser.add_argument("--sweep-span-pm", type=float, default=3.125, help="Размах одного пилообразного свипа в pm")
    parser.add_argument("--sweep-index", type=int, default=-1, help="Индекс референсного интервала свипа; -1 означает последний полный интервал")
    parser.add_argument("--post-start-s", type=float, default=None, help="Начало post-sweep анализа; по умолчанию конец выбранного свипа")
    parser.add_argument("--post-end-s", type=float, default=5.5, help="Конец post-sweep анализа")
    parser.add_argument("--initial-window-traces", type=int, default=50, help="Начальные same-parity трассы, используемые как ноль дрейфа")
    parser.add_argument("--tracking-step-sigma-pm", type=float, default=0.03, help="Масштаб непрерывности для tracking-а по банку референсов")
    parser.add_argument("--rolling-window", type=int, default=128, help="Окно rolling average для графиков")
    parser.add_argument("--block-rows", type=int, default=512, help="Число строк в одном корреляционном блоке")
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float64)
    distance_axis_m, _, _ = distance_axis_from_sampling_rate(
        result["real_segment_size"],
        result["sampling_rate"],
    )
    data, _, _, baseline_start_m, baseline_end_m = subtract_trace_baseline_from_tail(
        data,
        distance_axis_m,
        args.baseline_tail_m,
    )
    fiber_mask = (distance_axis_m >= float(args.fiber_z_min)) & (distance_axis_m <= float(args.fiber_z_max))
    if not np.any(fiber_mask):
        raise ValueError("Fiber window is empty")

    period_s = 1e-3 * float(args.reset_period_ms)
    reset_times_s = build_periodic_reset_grid(
        anchor_time_s=args.reset_anchor_time_s,
        period_s=period_s,
        min_time_s=0.0,
        max_time_s=args.max_reset_time_s,
    )
    if reset_times_s.size < 2:
        raise ValueError("Need at least two reset times to form a reference sweep")
    sweep_count = reset_times_s.size - 1
    selected_sweep_index = int(args.sweep_index)
    if selected_sweep_index < 0:
        selected_sweep_index = sweep_count + selected_sweep_index
    if not (0 <= selected_sweep_index < sweep_count):
        raise ValueError(f"sweep_index is out of range for {sweep_count} intervals")
    sweep_start_s = float(reset_times_s[selected_sweep_index])
    sweep_end_s = float(reset_times_s[selected_sweep_index + 1])
    post_start_s = sweep_end_s if args.post_start_s is None else float(args.post_start_s)

    parity_results = {}
    for parity in ["even", "odd"]:
        parity_data, parity_global_indices = select_parity_subset(data[:, fiber_mask], parity)
        parity_time_s = parity_global_indices.astype(np.float64) / float(result["scan_rate"])
        reference_mask = (parity_time_s >= sweep_start_s) & (parity_time_s < sweep_end_s)
        post_mask = (parity_time_s >= post_start_s) & (parity_time_s <= float(args.post_end_s))
        reference = parity_data[reference_mask]
        reference_time_s = parity_time_s[reference_mask]
        observed = parity_data[post_mask]
        post_time_s = parity_time_s[post_mask]
        if reference.shape[0] < 4:
            raise ValueError(f"Too few reference traces for parity {parity}")
        if observed.shape[0] < 1:
            raise ValueError(f"No post traces for parity {parity}")

        reference_lambda_pm = float(args.sweep_span_pm) * (reference_time_s - sweep_start_s) / (sweep_end_s - sweep_start_s)
        reference_norm = center_and_rms_normalize_rows(reference)
        observed_norm = center_and_rms_normalize_rows(observed)
        corr = correlation_matrix_blocked(observed_norm, reference_norm, args.block_rows)
        raw_index = np.nanargmax(corr, axis=1)
        raw_lambda_pm = reference_lambda_pm[raw_index]
        raw_corr = corr[np.arange(corr.shape[0]), raw_index]
        tracked_lambda_pm, tracked_corr, tracked_index = track_continuous_reference(
            corr,
            reference_lambda_pm,
            initial_window_traces=args.initial_window_traces,
            step_sigma_pm=args.tracking_step_sigma_pm,
        )
        zero_count = max(1, min(int(args.initial_window_traces), tracked_lambda_pm.size))
        lambda_zero_pm = float(np.nanmedian(tracked_lambda_pm[:zero_count]))
        parity_results[parity] = {
            "time_abs_s": post_time_s,
            "time_rel_s": post_time_s - post_start_s,
            "lambda_pm": tracked_lambda_pm,
            "lambda_pm_raw": raw_lambda_pm,
            "lambda_zero_pm": lambda_zero_pm,
            "drift_pm": tracked_lambda_pm - lambda_zero_pm,
            "fit_corr": tracked_corr,
            "fit_corr_raw": raw_corr,
            "tracked_index": tracked_index,
            "reference_time_s": reference_time_s,
            "reference_lambda_pm": reference_lambda_pm,
        }

    merged_time_abs_s = np.concatenate([parity_results["even"]["time_abs_s"], parity_results["odd"]["time_abs_s"]])
    merged_time_rel_s = np.concatenate([parity_results["even"]["time_rel_s"], parity_results["odd"]["time_rel_s"]])
    merged_lambda_pm = np.concatenate([parity_results["even"]["lambda_pm"], parity_results["odd"]["lambda_pm"]])
    merged_drift_pm = np.concatenate([parity_results["even"]["drift_pm"], parity_results["odd"]["drift_pm"]])
    merged_fit_corr = np.concatenate([parity_results["even"]["fit_corr"], parity_results["odd"]["fit_corr"]])
    order = np.argsort(merged_time_abs_s)
    merged_time_abs_s = merged_time_abs_s[order]
    merged_time_rel_s = merged_time_rel_s[order]
    merged_lambda_pm = merged_lambda_pm[order]
    merged_drift_pm = merged_drift_pm[order]
    merged_fit_corr = merged_fit_corr[order]

    merged_drift_pm_rolling = moving_average_ignore_nan(merged_drift_pm, args.rolling_window)
    merged_fit_corr_rolling = moving_average_ignore_nan(merged_fit_corr, args.rolling_window)

    suffix = "wavelength_drift_trace_bank_after_last_sweep"
    fig1, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax1.plot(merged_time_abs_s, merged_lambda_pm, ".", color="#B0B0B0", markersize=2.0, alpha=0.45, label="Объединённые")
    ax1.plot(parity_results["even"]["time_abs_s"], parity_results["even"]["lambda_pm"], ".", color="#1F77B4", markersize=1.8, alpha=0.55, label="Чётные")
    ax1.plot(parity_results["odd"]["time_abs_s"], parity_results["odd"]["lambda_pm"], ".", color="#FF7F0E", markersize=1.8, alpha=0.55, label="Нечётные")
    ax1.axvline(sweep_end_s, color="#D62728", linewidth=1.0, alpha=0.8, label="Конец последнего свипа")
    ax1.set_xlabel("Время (s)")
    ax1.set_ylabel("Лучшая координата внутри последнего свипа (pm)")
    ax1.set_title("Координата длины волны после выключения модуляции лазера")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")
    lambda_png_path = output_dir / f"{dat_path.stem}_{suffix}.png"
    fig1.savefig(lambda_png_path, dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax2.plot(merged_time_abs_s, merged_drift_pm, ".", color="#808080", markersize=2.0, alpha=0.45, label="Объединённые raw")
    ax2.plot(merged_time_abs_s, merged_drift_pm_rolling, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax2.axvline(sweep_end_s, color="#D62728", linewidth=1.0, alpha=0.8, label="Конец последнего свипа")
    ax2.set_xlabel("Время (s)")
    ax2.set_ylabel("Дрейф от начального post-sweep уровня (pm)")
    ax2.set_title("Дрейф длины волны по сопоставлению с банком трасс")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")
    drift_png_path = output_dir / f"{dat_path.stem}_{suffix}_drift.png"
    fig2.savefig(drift_png_path, dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax3.plot(merged_time_abs_s, merged_fit_corr, ".", color="#4C78A8", markersize=2.0, alpha=0.55, label="Сырые точки")
    ax3.plot(merged_time_abs_s, merged_fit_corr_rolling, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax3.axvline(sweep_end_s, color="#D62728", linewidth=1.0, alpha=0.8, label="Конец последнего свипа")
    ax3.set_xlabel("Время (s)")
    ax3.set_ylabel("Лучшая корреляция")
    ax3.set_title("Качество fit-а по банку трасс")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")
    quality_png_path = output_dir / f"{dat_path.stem}_{suffix}_fit_quality.png"
    fig3.savefig(quality_png_path, dpi=200)
    plt.close(fig3)

    csv_path = output_dir / f"{dat_path.stem}_{suffix}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("time_abs_s,time_rel_s,lambda_pm,drift_pm,fit_corr\n")
        for row in zip(merged_time_abs_s, merged_time_rel_s, merged_lambda_pm, merged_drift_pm, merged_fit_corr):
            fout.write(f"{row[0]:.10f},{row[1]:.10f},{row[2]:.10f},{row[3]:.10f},{row[4]:.10f}\n")

    mat_path, script_path = save_matlab_bundle(
        output_dir,
        dat_path.stem,
        suffix,
        {
            "fiber_distance_m": distance_axis_m[fiber_mask][:, None],
            "reset_times_s": reset_times_s[:, None],
            "selected_sweep_index": np.array([[selected_sweep_index]], dtype=np.int32),
            "sweep_start_s": np.array([[sweep_start_s]], dtype=np.float64),
            "sweep_end_s": np.array([[sweep_end_s]], dtype=np.float64),
            "post_start_s": np.array([[post_start_s]], dtype=np.float64),
            "post_end_s": np.array([[args.post_end_s]], dtype=np.float64),
            "sweep_span_pm": np.array([[args.sweep_span_pm]], dtype=np.float64),
            "baseline_window_start_m": np.array([[baseline_start_m]], dtype=np.float64),
            "baseline_window_end_m": np.array([[baseline_end_m]], dtype=np.float64),
            "even_time_abs_s": parity_results["even"]["time_abs_s"][:, None],
            "odd_time_abs_s": parity_results["odd"]["time_abs_s"][:, None],
            "even_lambda_pm": parity_results["even"]["lambda_pm"][:, None],
            "odd_lambda_pm": parity_results["odd"]["lambda_pm"][:, None],
            "even_drift_pm": parity_results["even"]["drift_pm"][:, None],
            "odd_drift_pm": parity_results["odd"]["drift_pm"][:, None],
            "even_fit_corr": parity_results["even"]["fit_corr"][:, None],
            "odd_fit_corr": parity_results["odd"]["fit_corr"][:, None],
            "even_reference_lambda_pm": parity_results["even"]["reference_lambda_pm"][:, None],
            "odd_reference_lambda_pm": parity_results["odd"]["reference_lambda_pm"][:, None],
            "merged_time_abs_s": merged_time_abs_s[:, None],
            "merged_time_rel_s": merged_time_rel_s[:, None],
            "merged_lambda_pm": merged_lambda_pm[:, None],
            "merged_drift_pm": merged_drift_pm[:, None],
            "merged_drift_pm_rolling": merged_drift_pm_rolling[:, None],
            "merged_fit_corr": merged_fit_corr[:, None],
            "merged_fit_corr_rolling": merged_fit_corr_rolling[:, None],
        },
    )

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"fiber_start_m: {distance_axis_m[fiber_mask][0]:.6f}")
    print(f"fiber_end_m: {distance_axis_m[fiber_mask][-1]:.6f}")
    print(f"selected_sweep_index: {selected_sweep_index}")
    print(f"sweep_start_s: {sweep_start_s:.10f}")
    print(f"sweep_end_s: {sweep_end_s:.10f}")
    print(f"post_start_s: {post_start_s:.10f}")
    print(f"post_end_s: {float(args.post_end_s):.10f}")
    print(f"period_s: {period_s:.10f}")
    print(f"sweep_span_pm: {args.sweep_span_pm}")
    print(f"even_reference_count: {parity_results['even']['reference_time_s'].size}")
    print(f"odd_reference_count: {parity_results['odd']['reference_time_s'].size}")
    print(f"merged_trace_count: {merged_time_abs_s.size}")
    print(f"even_lambda_zero_pm: {parity_results['even']['lambda_zero_pm']:.10f}")
    print(f"odd_lambda_zero_pm: {parity_results['odd']['lambda_zero_pm']:.10f}")
    print(f"drift_pm_start: {merged_drift_pm[0]:.10f}")
    print(f"drift_pm_end: {merged_drift_pm[-1]:.10f}")
    print(f"drift_pm_min: {np.nanmin(merged_drift_pm):.10f}")
    print(f"drift_pm_max: {np.nanmax(merged_drift_pm):.10f}")
    print(f"fit_corr_mean: {np.nanmean(merged_fit_corr):.10f}")
    print(f"fit_corr_min: {np.nanmin(merged_fit_corr):.10f}")
    print(f"lambda_png_saved_to: {lambda_png_path}")
    print(f"drift_png_saved_to: {drift_png_path}")
    print(f"fit_quality_png_saved_to: {quality_png_path}")
    print(f"csv_saved_to: {csv_path}")
    print(f"matlab_data_saved_to: {mat_path}")
    print(f"matlab_script_saved_to: {script_path}")


if __name__ == "__main__":
    main()
