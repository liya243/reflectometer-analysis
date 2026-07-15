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


def block_average_rows(data, time_s, block_size):
    data = np.asarray(data, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    block_size = int(block_size)
    if block_size <= 1:
        return data.copy(), time_s.copy()
    block_count = data.shape[0] // block_size
    if block_count == 0:
        return data.copy(), time_s.copy()
    kept = block_count * block_size
    averaged = data[:kept].reshape(block_count, block_size, data.shape[1]).mean(axis=1)
    averaged_time_s = time_s[:kept].reshape(block_count, block_size).mean(axis=1)
    return averaged, averaged_time_s


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


def estimate_initial_lambda(reference_norm, reference_lambda_pm, initial_norm):
    corr = initial_norm @ reference_norm.T / float(reference_norm.shape[1])
    score = np.nanmean(corr, axis=0)
    best_idx = int(np.nanargmax(score))
    return float(reference_lambda_pm[best_idx]), best_idx, score


def fit_local_slope(reference_norm, reference_lambda_pm, lambda0_pm, half_width_pm, min_points):
    reference_lambda_pm = np.asarray(reference_lambda_pm, dtype=np.float64)
    local = np.abs(reference_lambda_pm - float(lambda0_pm)) <= float(half_width_pm)
    if np.count_nonzero(local) < int(min_points):
        order = np.argsort(np.abs(reference_lambda_pm - float(lambda0_pm)))
        local = np.zeros(reference_lambda_pm.size, dtype=bool)
        local[order[: int(min_points)]] = True

    x = reference_lambda_pm[local] - float(lambda0_pm)
    y = reference_norm[local]
    design = np.column_stack([np.ones_like(x), x])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = coeff[0]
    slope = coeff[1]
    return intercept, slope, local


def estimate_signed_drift(observed_norm, intercept, slope, slope_percentile):
    slope = np.asarray(slope, dtype=np.float64).reshape(-1)
    intercept = np.asarray(intercept, dtype=np.float64).reshape(-1)
    observed_norm = np.asarray(observed_norm, dtype=np.float64)

    slope_abs = np.abs(slope)
    threshold = np.percentile(slope_abs[np.isfinite(slope_abs)], float(slope_percentile))
    coord_mask = np.isfinite(slope) & (slope_abs >= threshold)
    if np.count_nonzero(coord_mask) < 4:
        coord_mask = np.isfinite(slope)

    s = slope[coord_mask]
    baseline = intercept[coord_mask]
    obs = observed_norm[:, coord_mask]
    denom = float(np.dot(s, s))
    if denom <= 1e-18:
        raise ValueError("Local wavelength slope is too small")

    delta_pm = (obs - baseline[None, :]) @ s / denom
    model = baseline[None, :] + delta_pm[:, None] * s[None, :]
    obs_n = center_and_rms_normalize_rows(obs)
    model_n = center_and_rms_normalize_rows(model)
    fit_corr = np.sum(obs_n * model_n, axis=1) / float(obs_n.shape[1])
    return delta_pm, fit_corr, coord_mask


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)
    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

figure('Color', 'w', 'Name', 'Знаковый дрейф длины волны');
plot(data.merged_time_abs_s, data.merged_delta_lambda_pm, '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 8);
hold on;
plot(data.merged_time_abs_s, data.merged_delta_lambda_pm_rolling, 'k', 'LineWidth', 1.6);
plot(data.even_time_abs_s, data.even_delta_lambda_pm, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 7);
plot(data.odd_time_abs_s, data.odd_delta_lambda_pm, '.', 'Color', [1.00 0.50 0.05], 'MarkerSize', 7);
grid on; xlabel('Время (s)'); ylabel('Знаковый \\Delta\\lambda (pm)');
title('Знаковый дрейф по local-slope последнего свипа');
legend('Объединённые raw', 'Объединённые rolling', 'Чётные', 'Нечётные', 'Location', 'best');

figure('Color', 'w', 'Name', 'Качество fit-а');
plot(data.merged_time_abs_s, data.merged_fit_corr, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 8);
hold on;
plot(data.merged_time_abs_s, data.merged_fit_corr_rolling, 'k', 'LineWidth', 1.6);
grid on; xlabel('Время (s)'); ylabel('Корреляция fit-а');
title('Качество local-slope fit-а');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return mat_path, script_path


def main():
    parser = argparse.ArgumentParser(
        description="Оценить знаковый дрейф длины волны после свипа по локальной производной, измеренной на последнем свипе."
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
    parser.add_argument("--sweep-index", type=int, default=-1, help="Индекс свипа, используемого как референс локальной производной")
    parser.add_argument("--post-start-s", type=float, default=None, help="Начало post-sweep анализа; по умолчанию конец выбранного свипа")
    parser.add_argument("--post-end-s", type=float, default=5.5, help="Конец post-sweep анализа")
    parser.add_argument("--block-size", type=int, default=64, help="Усреднять столько same-parity post-трасс на одну оценку")
    parser.add_argument("--initial-window-blocks", type=int, default=4, help="Число первых post-блоков для выбора lambda0 и нуля")
    parser.add_argument("--local-half-width-pm", type=float, default=0.35, help="Полуширина окна вокруг lambda0 для fit-а dTrace/dlambda")
    parser.add_argument("--local-min-points", type=int, default=40, help="Минимальное число референсных трасс свипа для локального slope")
    parser.add_argument("--slope-percentile", type=float, default=50.0, help="Использовать координаты, где |slope| выше этого процентиля")
    parser.add_argument("--rolling-window", type=int, default=9, help="Окно rolling average по блочным оценкам")
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
    fiber_distance_m = distance_axis_m[fiber_mask]

    period_s = 1e-3 * float(args.reset_period_ms)
    reset_times_s = build_periodic_reset_grid(
        anchor_time_s=args.reset_anchor_time_s,
        period_s=period_s,
        min_time_s=0.0,
        max_time_s=args.max_reset_time_s,
    )
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
        post = parity_data[post_mask]
        post_time_s = parity_time_s[post_mask]
        if reference.shape[0] < int(args.local_min_points):
            raise ValueError(f"Too few reference traces for parity {parity}")
        if post.shape[0] == 0:
            raise ValueError(f"No post traces for parity {parity}")

        post_avg, post_time_avg_s = block_average_rows(post, post_time_s, args.block_size)
        reference_lambda_pm = float(args.sweep_span_pm) * (reference_time_s - sweep_start_s) / (sweep_end_s - sweep_start_s)
        reference_norm = center_and_rms_normalize_rows(reference)
        post_norm = center_and_rms_normalize_rows(post_avg)

        initial_count = max(1, min(int(args.initial_window_blocks), post_norm.shape[0]))
        lambda0_pm, lambda0_index, lambda_score = estimate_initial_lambda(
            reference_norm,
            reference_lambda_pm,
            post_norm[:initial_count],
        )
        intercept, slope, local_mask = fit_local_slope(
            reference_norm,
            reference_lambda_pm,
            lambda0_pm=lambda0_pm,
            half_width_pm=args.local_half_width_pm,
            min_points=args.local_min_points,
        )
        delta_pm, fit_corr, coord_mask = estimate_signed_drift(
            post_norm,
            intercept,
            slope,
            slope_percentile=args.slope_percentile,
        )
        zero_pm = float(np.nanmedian(delta_pm[:initial_count]))
        parity_results[parity] = {
            "time_abs_s": post_time_avg_s,
            "time_rel_s": post_time_avg_s - post_start_s,
            "delta_lambda_pm": delta_pm - zero_pm,
            "delta_lambda_raw_pm": delta_pm,
            "fit_corr": fit_corr,
            "lambda0_pm": lambda0_pm,
            "lambda0_index": lambda0_index,
            "lambda_score": lambda_score,
            "reference_lambda_pm": reference_lambda_pm,
            "local_mask": local_mask,
            "coord_mask": coord_mask,
        }

    merged_time_abs_s = np.concatenate([parity_results["even"]["time_abs_s"], parity_results["odd"]["time_abs_s"]])
    merged_time_rel_s = np.concatenate([parity_results["even"]["time_rel_s"], parity_results["odd"]["time_rel_s"]])
    merged_delta_lambda_pm = np.concatenate([parity_results["even"]["delta_lambda_pm"], parity_results["odd"]["delta_lambda_pm"]])
    merged_fit_corr = np.concatenate([parity_results["even"]["fit_corr"], parity_results["odd"]["fit_corr"]])
    order = np.argsort(merged_time_abs_s)
    merged_time_abs_s = merged_time_abs_s[order]
    merged_time_rel_s = merged_time_rel_s[order]
    merged_delta_lambda_pm = merged_delta_lambda_pm[order]
    merged_fit_corr = merged_fit_corr[order]
    merged_delta_lambda_pm_rolling = moving_average_ignore_nan(merged_delta_lambda_pm, args.rolling_window)
    merged_fit_corr_rolling = moving_average_ignore_nan(merged_fit_corr, args.rolling_window)

    suffix = "wavelength_drift_local_slope_after_last_sweep"
    fig1, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax1.plot(merged_time_abs_s, merged_delta_lambda_pm, ".", color="#A8A8A8", markersize=5, alpha=0.55, label="Объединённые raw")
    ax1.plot(merged_time_abs_s, merged_delta_lambda_pm_rolling, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax1.plot(parity_results["even"]["time_abs_s"], parity_results["even"]["delta_lambda_pm"], ".", color="#1F77B4", markersize=4, alpha=0.70, label="Чётные")
    ax1.plot(parity_results["odd"]["time_abs_s"], parity_results["odd"]["delta_lambda_pm"], ".", color="#FF7F0E", markersize=4, alpha=0.70, label="Нечётные")
    ax1.axvline(sweep_end_s, color="#D62728", linewidth=1.0, alpha=0.8, label="Конец последнего свипа")
    ax1.set_xlabel("Время (s)")
    ax1.set_ylabel("Знаковый дрейф длины волны (pm)")
    ax1.set_title("Знаковый дрейф длины волны по локальному slope последнего свипа")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")
    drift_png_path = output_dir / f"{dat_path.stem}_{suffix}.png"
    fig1.savefig(drift_png_path, dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax2.plot(merged_time_abs_s, merged_fit_corr, ".", color="#4C78A8", markersize=5, alpha=0.65, label="Сырые точки")
    ax2.plot(merged_time_abs_s, merged_fit_corr_rolling, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax2.axvline(sweep_end_s, color="#D62728", linewidth=1.0, alpha=0.8, label="Конец последнего свипа")
    ax2.set_xlabel("Время (s)")
    ax2.set_ylabel("Корреляция линейного fit-а")
    ax2.set_title("Качество local-slope fit-а")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")
    quality_png_path = output_dir / f"{dat_path.stem}_{suffix}_fit_quality.png"
    fig2.savefig(quality_png_path, dpi=200)
    plt.close(fig2)

    csv_path = output_dir / f"{dat_path.stem}_{suffix}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("time_abs_s,time_rel_s,delta_lambda_pm,fit_corr\n")
        for row in zip(merged_time_abs_s, merged_time_rel_s, merged_delta_lambda_pm, merged_fit_corr):
            fout.write(f"{row[0]:.10f},{row[1]:.10f},{row[2]:.10f},{row[3]:.10f}\n")

    mat_path, script_path = save_matlab_bundle(
        output_dir,
        dat_path.stem,
        suffix,
        {
            "fiber_distance_m": fiber_distance_m[:, None],
            "selected_sweep_index": np.array([[selected_sweep_index]], dtype=np.int32),
            "sweep_start_s": np.array([[sweep_start_s]], dtype=np.float64),
            "sweep_end_s": np.array([[sweep_end_s]], dtype=np.float64),
            "post_start_s": np.array([[post_start_s]], dtype=np.float64),
            "post_end_s": np.array([[args.post_end_s]], dtype=np.float64),
            "sweep_span_pm": np.array([[args.sweep_span_pm]], dtype=np.float64),
            "baseline_window_start_m": np.array([[baseline_start_m]], dtype=np.float64),
            "baseline_window_end_m": np.array([[baseline_end_m]], dtype=np.float64),
            "even_lambda0_pm": np.array([[parity_results["even"]["lambda0_pm"]]], dtype=np.float64),
            "odd_lambda0_pm": np.array([[parity_results["odd"]["lambda0_pm"]]], dtype=np.float64),
            "even_time_abs_s": parity_results["even"]["time_abs_s"][:, None],
            "odd_time_abs_s": parity_results["odd"]["time_abs_s"][:, None],
            "even_delta_lambda_pm": parity_results["even"]["delta_lambda_pm"][:, None],
            "odd_delta_lambda_pm": parity_results["odd"]["delta_lambda_pm"][:, None],
            "even_fit_corr": parity_results["even"]["fit_corr"][:, None],
            "odd_fit_corr": parity_results["odd"]["fit_corr"][:, None],
            "merged_time_abs_s": merged_time_abs_s[:, None],
            "merged_time_rel_s": merged_time_rel_s[:, None],
            "merged_delta_lambda_pm": merged_delta_lambda_pm[:, None],
            "merged_delta_lambda_pm_rolling": merged_delta_lambda_pm_rolling[:, None],
            "merged_fit_corr": merged_fit_corr[:, None],
            "merged_fit_corr_rolling": merged_fit_corr_rolling[:, None],
        },
    )

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"selected_sweep_index: {selected_sweep_index}")
    print(f"sweep_start_s: {sweep_start_s:.10f}")
    print(f"sweep_end_s: {sweep_end_s:.10f}")
    print(f"post_start_s: {post_start_s:.10f}")
    print(f"post_end_s: {float(args.post_end_s):.10f}")
    print(f"block_size: {args.block_size}")
    print(f"even_lambda0_pm: {parity_results['even']['lambda0_pm']:.10f}")
    print(f"odd_lambda0_pm: {parity_results['odd']['lambda0_pm']:.10f}")
    print(f"merged_block_count: {merged_delta_lambda_pm.size}")
    print(f"delta_lambda_pm_start: {merged_delta_lambda_pm[0]:.10f}")
    print(f"delta_lambda_pm_end: {merged_delta_lambda_pm[-1]:.10f}")
    print(f"delta_lambda_pm_min: {np.nanmin(merged_delta_lambda_pm):.10f}")
    print(f"delta_lambda_pm_max: {np.nanmax(merged_delta_lambda_pm):.10f}")
    print(f"fit_corr_mean: {np.nanmean(merged_fit_corr):.10f}")
    print(f"fit_corr_min: {np.nanmin(merged_fit_corr):.10f}")
    print(f"drift_png_saved_to: {drift_png_path}")
    print(f"fit_quality_png_saved_to: {quality_png_path}")
    print(f"csv_saved_to: {csv_path}")
    print(f"matlab_data_saved_to: {mat_path}")
    print(f"matlab_script_saved_to: {script_path}")


if __name__ == "__main__":
    main()
