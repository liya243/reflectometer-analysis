import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

from analysis_output_utils import cleanup_outputs_for_dataset, matlab_safe_stem
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


def reconstruct_modeled_trace_bank(harmonics, lag_indices, delta_lambda_grid_pm, n_eff, lambda0_nm, distance_step_m):
    harmonics = np.asarray(harmonics, dtype=np.complex128)
    lag_indices = np.asarray(lag_indices, dtype=np.int64).reshape(-1)
    lambda0_m = float(lambda0_nm) * 1e-9
    delta_lambda_m = np.asarray(delta_lambda_grid_pm, dtype=np.float64) * 1e-12
    delta_beta = -2.0 * np.pi * float(n_eff) * delta_lambda_m / (lambda0_m**2)
    lag_distances_m = lag_indices.astype(np.float64) * float(distance_step_m)
    phase_matrix = np.exp(1j * 2.0 * np.outer(lag_distances_m, delta_beta))
    modeled = 2.0 * np.real(harmonics @ phase_matrix)
    return modeled.T


def estimate_delta_lambda_pm(observed_traces, modeled_trace_bank, delta_lambda_grid_pm):
    observed_norm = center_and_rms_normalize_rows(observed_traces)
    modeled_norm = center_and_rms_normalize_rows(modeled_trace_bank)
    correlation = observed_norm @ modeled_norm.T / float(observed_norm.shape[1])
    best_index = np.nanargmax(correlation, axis=1)
    best_delta_lambda_pm = np.asarray(delta_lambda_grid_pm, dtype=np.float64)[best_index]
    best_correlation = correlation[np.arange(correlation.shape[0]), best_index]
    return best_delta_lambda_pm, best_correlation, correlation


def track_continuous_lambda(correlation_matrix, delta_lambda_grid_pm, initial_window_traces, step_sigma_pm):
    correlation_matrix = np.asarray(correlation_matrix, dtype=np.float64)
    delta_lambda_grid_pm = np.asarray(delta_lambda_grid_pm, dtype=np.float64).reshape(-1)
    if correlation_matrix.ndim != 2:
        raise ValueError("correlation_matrix must be 2D")
    if correlation_matrix.shape[1] != delta_lambda_grid_pm.size:
        raise ValueError("Grid size mismatch")

    initial_count = max(1, min(int(initial_window_traces), correlation_matrix.shape[0]))
    initial_score = np.mean(correlation_matrix[:initial_count], axis=0)
    tracked_index = np.empty(correlation_matrix.shape[0], dtype=np.int64)
    tracked_index[0] = int(np.nanargmax(initial_score))

    grid = delta_lambda_grid_pm
    inv_two_sigma2 = 0.5 / max(float(step_sigma_pm) ** 2, 1e-12)
    for row in range(1, correlation_matrix.shape[0]):
        prev_lambda = grid[tracked_index[row - 1]]
        penalty = inv_two_sigma2 * (grid - prev_lambda) ** 2
        score = correlation_matrix[row] - penalty
        tracked_index[row] = int(np.nanargmax(score))

    tracked_lambda_pm = grid[tracked_index]
    tracked_corr = correlation_matrix[np.arange(correlation_matrix.shape[0]), tracked_index]
    return tracked_lambda_pm, tracked_corr, tracked_index


def refine_subgrid_peak(correlation_matrix, delta_lambda_grid_pm, tracked_index):
    correlation_matrix = np.asarray(correlation_matrix, dtype=np.float64)
    delta_lambda_grid_pm = np.asarray(delta_lambda_grid_pm, dtype=np.float64).reshape(-1)
    tracked_index = np.asarray(tracked_index, dtype=np.int64).reshape(-1)
    refined_lambda_pm = delta_lambda_grid_pm[tracked_index].astype(np.float64).copy()
    refined_corr = correlation_matrix[np.arange(correlation_matrix.shape[0]), tracked_index].astype(np.float64).copy()
    if delta_lambda_grid_pm.size < 3:
        return refined_lambda_pm, refined_corr

    grid_step = float(np.median(np.diff(delta_lambda_grid_pm)))
    for row, k in enumerate(tracked_index):
        if not (0 < k < delta_lambda_grid_pm.size - 1):
            continue
        y0 = correlation_matrix[row, k - 1]
        y1 = correlation_matrix[row, k]
        y2 = correlation_matrix[row, k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-12:
            continue
        shift = 0.5 * (y0 - y2) / denom
        shift = float(np.clip(shift, -1.0, 1.0))
        refined_lambda_pm[row] = delta_lambda_grid_pm[k] + shift * grid_step
        refined_corr[row] = y1 - 0.25 * (y0 - y2) * shift
    return refined_lambda_pm, refined_corr


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Wavelength after last sweep');
plot(data.merged_time_rel_s, data.merged_lambda_pm, '.', 'Color', [0.55 0.55 0.55], 'MarkerSize', 5);
hold on;
plot(data.merged_time_rel_s, data.merged_lambda_pm_rolling, 'k', 'LineWidth', 1.6);
plot(data.even_time_rel_s, data.even_lambda_pm, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 4);
plot(data.odd_time_rel_s, data.odd_lambda_pm, '.', 'Color', [1.00 0.50 0.05], 'MarkerSize', 4);
grid on;
xlabel('Time after last sweep (s)');
ylabel('\\Delta\\lambda fitted from last sweep model (pm)');
title('Estimated wavelength after last sweep');
legend('Merged raw', 'Merged rolling', 'Even', 'Odd', 'Location', 'best');

f2 = figure('Color', 'w', 'Name', 'Wavelength drift after last sweep');
plot(data.merged_time_rel_s, data.merged_drift_pm, '.', 'Color', [0.55 0.55 0.55], 'MarkerSize', 5);
hold on;
plot(data.merged_time_rel_s, data.merged_drift_pm_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Time after last sweep (s)');
ylabel('Drift relative to first post-sweep trace (pm)');
title('Wavelength drift after last sweep');

f3 = figure('Color', 'w', 'Name', 'Fit correlation');
plot(data.merged_time_rel_s, data.merged_fit_corr, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 5);
hold on;
plot(data.merged_time_rel_s, data.merged_fit_corr_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Time after last sweep (s)');
ylabel('Best model correlation');
title('Fit quality after last sweep');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return mat_path, script_path


def main():
    parser = argparse.ArgumentParser(
        description="Estimate wavelength drift during the second after the last modulation sweep using the recovered last-sweep harmonic model."
    )
    parser.add_argument("dat_path", help="Path to the .dat file")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Directory for output files")
    parser.add_argument("--model-mat", default=None, help="MAT file produced by solve_complex_amplitudes_from_harmonics.py")
    parser.add_argument("--scan-rate", type=float, default=None, help="Optional override for reflectogram scan rate")
    parser.add_argument("--fiber-z-min", type=float, default=110.0, help="Start of real fiber region in meters")
    parser.add_argument("--fiber-z-max", type=float, default=350.0, help="End of real fiber region in meters")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Subtract per-trace baseline from the last this many meters")
    parser.add_argument("--lambda0-nm", type=float, default=1550.0, help="Central wavelength in nm")
    parser.add_argument("--rolling-window", type=int, default=128, help="Rolling window for plotted drift curves")
    parser.add_argument("--min-period-s", type=float, default=0.03, help="Minimum sweep period for reset detection")
    parser.add_argument("--max-period-s", type=float, default=0.2, help="Maximum sweep period for reset detection")
    parser.add_argument("--prominence-sigma", type=float, default=2.0, help="Reset detector threshold in robust sigma units")
    parser.add_argument("--refine-window-fraction", type=float, default=0.15, help="Local refinement window as fraction of detected period")
    parser.add_argument("--reset-time-shift-ms", type=float, default=0.0, help="Shift detected reset times later by this many milliseconds")
    parser.add_argument("--reset-period-override-ms", type=float, default=None, help="Use this fixed reset period in milliseconds for sweep boundaries")
    parser.add_argument("--reset-anchor-time-s", type=float, default=None, help="Anchor time for fixed reset grid; grid is extended backward and forward")
    parser.add_argument("--max-reset-time-s", type=float, default=4.6, help="Ignore resets after this time")
    parser.add_argument("--reset-detection-end-time-s", type=float, default=4.6, help="Use only traces up to this time when detecting last sweep")
    parser.add_argument("--post-duration-s", type=float, default=1.0, help="How much time after the last sweep to analyze")
    parser.add_argument("--lambda-grid-min-pm", type=float, default=-0.5, help="Minimum fitted wavelength shift in pm")
    parser.add_argument("--lambda-grid-max-pm", type=float, default=4.0, help="Maximum fitted wavelength shift in pm")
    parser.add_argument("--lambda-grid-step-pm", type=float, default=0.002, help="Step of the fitted wavelength grid in pm")
    parser.add_argument("--tracking-step-sigma-pm", type=float, default=0.01, help="Continuity penalty scale for branch tracking in pm per trace step")
    parser.add_argument("--tracking-initial-window-traces", type=int, default=50, help="Use this many first post-sweep traces to choose the initial wavelength branch")
    parser.add_argument("--cleanup-dataset-outputs", action="store_true", help="Delete old outputs for this dataset before saving new ones")
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    if args.cleanup_dataset_outputs:
        cleanup_outputs_for_dataset(output_dir, dat_path.stem)

    if args.model_mat is None:
        model_mat_path = output_dir / f"{dat_path.stem}_complex_amplitude_factorization_single_sweep_matlab_data.mat"
    else:
        model_mat_path = Path(args.model_mat)
    model = loadmat(model_mat_path)

    even_harmonics = np.asarray(model["even_direct_modeled_harmonics"], dtype=np.complex128)
    odd_harmonics = np.asarray(model["odd_direct_modeled_harmonics"], dtype=np.complex128)
    fit_lag_indices = np.asarray(model["fit_lag_indices"], dtype=np.int64).reshape(-1)
    selected_sweep_index = int(np.asarray(model["sweep_index"]).reshape(-1)[0])
    distance_step_m = float(np.asarray(model["distance_step_m"]).reshape(-1)[0])

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float64)
    distance_axis_m, _, n_eff = distance_axis_from_sampling_rate(
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

    shared_global_indices = np.arange(data.shape[0], dtype=np.int64)
    shared_data = data[:, fiber_mask]
    keep = shared_global_indices.astype(np.float64) / float(result["scan_rate"]) <= float(args.reset_detection_end_time_s)
    reset_times_s, dominant_period_s = detect_reset_times(
        shared_data[keep],
        shared_global_indices[keep],
        result["scan_rate"],
        rolling_window=args.rolling_window,
        min_period_s=args.min_period_s,
        prominence_sigma=args.prominence_sigma,
        refine_window_fraction=args.refine_window_fraction,
        max_period_s=args.max_period_s,
    )
    reset_times_s = reset_times_s + 1e-3 * float(args.reset_time_shift_ms)
    reset_times_s = reset_times_s[reset_times_s <= float(args.max_reset_time_s)]
    reset_times_s = regularize_reset_grid(reset_times_s, dominant_period_s)
    if args.reset_period_override_ms is not None:
        override_period_s = 1e-3 * float(args.reset_period_override_ms)
        if args.reset_anchor_time_s is None:
            if reset_times_s.size == 0:
                raise ValueError("Cannot infer reset anchor from an empty reset list")
            anchor_time_s = float(reset_times_s[0])
        else:
            anchor_time_s = float(args.reset_anchor_time_s)
        reset_times_s = build_periodic_reset_grid(
            anchor_time_s=anchor_time_s,
            period_s=override_period_s,
            min_time_s=0.0,
            max_time_s=float(args.max_reset_time_s),
        )
        dominant_period_s = override_period_s
    if reset_times_s.size < 2:
        raise ValueError("Not enough reset times to define the last sweep")
    if not (0 <= selected_sweep_index + 1 < reset_times_s.size):
        raise ValueError("Stored sweep_index is inconsistent with detected reset times")
    last_sweep_end_s = float(reset_times_s[selected_sweep_index + 1])

    delta_lambda_grid_pm = np.arange(
        float(args.lambda_grid_min_pm),
        float(args.lambda_grid_max_pm) + 0.5 * float(args.lambda_grid_step_pm),
        float(args.lambda_grid_step_pm),
        dtype=np.float64,
    )
    even_model_bank = reconstruct_modeled_trace_bank(
        even_harmonics,
        fit_lag_indices,
        delta_lambda_grid_pm,
        n_eff=n_eff,
        lambda0_nm=args.lambda0_nm,
        distance_step_m=distance_step_m,
    )
    odd_model_bank = reconstruct_modeled_trace_bank(
        odd_harmonics,
        fit_lag_indices,
        delta_lambda_grid_pm,
        n_eff=n_eff,
        lambda0_nm=args.lambda0_nm,
        distance_step_m=distance_step_m,
    )

    parity_results = {}
    for parity, model_bank in [("even", even_model_bank), ("odd", odd_model_bank)]:
        parity_data, parity_global_indices = select_parity_subset(data[:, fiber_mask], parity)
        parity_time_s = parity_global_indices.astype(np.float64) / float(result["scan_rate"])
        post_mask = (parity_time_s >= last_sweep_end_s) & (parity_time_s < last_sweep_end_s + float(args.post_duration_s))
        observed = parity_data[post_mask]
        post_time_s = parity_time_s[post_mask]
        if observed.shape[0] == 0:
            raise ValueError(f"No post-sweep traces remain for parity '{parity}'")
        raw_lambda_pm, raw_corr, corr_matrix = estimate_delta_lambda_pm(observed, model_bank, delta_lambda_grid_pm)
        tracked_lambda_pm, tracked_corr, tracked_index = track_continuous_lambda(
            corr_matrix,
            delta_lambda_grid_pm,
            initial_window_traces=args.tracking_initial_window_traces,
            step_sigma_pm=args.tracking_step_sigma_pm,
        )
        tracked_lambda_pm, tracked_corr = refine_subgrid_peak(
            corr_matrix,
            delta_lambda_grid_pm,
            tracked_index,
        )
        parity_results[parity] = {
            "time_abs_s": post_time_s,
            "time_rel_s": post_time_s - last_sweep_end_s,
            "lambda_pm_raw": raw_lambda_pm,
            "fit_corr_raw": raw_corr,
            "lambda_pm": tracked_lambda_pm,
            "drift_pm": tracked_lambda_pm - tracked_lambda_pm[0],
            "fit_corr": tracked_corr,
            "tracked_index": tracked_index,
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

    merged_lambda_pm_rolling = moving_average_ignore_nan(merged_lambda_pm, args.rolling_window)
    merged_drift_pm_rolling = moving_average_ignore_nan(merged_drift_pm, args.rolling_window)
    merged_fit_corr_rolling = moving_average_ignore_nan(merged_fit_corr, args.rolling_window)

    suffix = "wavelength_drift_after_last_sweep"

    fig1, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax1.plot(merged_time_rel_s, merged_lambda_pm, ".", color="#C7C7C7", markersize=2.0, alpha=0.30, label="Merged raw")
    ax1.plot(parity_results["even"]["time_rel_s"], parity_results["even"]["lambda_pm"], ".", color="#1F77B4", markersize=1.8, alpha=0.55, label="Even")
    ax1.plot(parity_results["odd"]["time_rel_s"], parity_results["odd"]["lambda_pm"], ".", color="#FF7F0E", markersize=1.8, alpha=0.55, label="Odd")
    ax1.set_xlabel("Time after last sweep (s)")
    ax1.set_ylabel("Fitted wavelength coordinate (pm)")
    ax1.set_title("Estimated wavelength after the last sweep, with parity branch ambiguity")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")
    lambda_png_path = output_dir / f"{dat_path.stem}_{suffix}.png"
    fig1.savefig(lambda_png_path, dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax2.plot(merged_time_rel_s, merged_drift_pm, ".", color="#7F7F7F", markersize=2.5, alpha=0.55, label="Merged raw")
    ax2.plot(merged_time_rel_s, merged_drift_pm_rolling, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax2.set_xlabel("Time after last sweep (s)")
    ax2.set_ylabel("Drift from first post-sweep trace (pm)")
    ax2.set_title("Wavelength drift during the second after the last sweep")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")
    drift_png_path = output_dir / f"{dat_path.stem}_{suffix}_drift.png"
    fig2.savefig(drift_png_path, dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax3.plot(merged_time_rel_s, merged_fit_corr, ".", color="#4C78A8", markersize=2.5, alpha=0.55, label="Raw")
    ax3.plot(merged_time_rel_s, merged_fit_corr_rolling, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax3.set_xlabel("Time after last sweep (s)")
    ax3.set_ylabel("Best model correlation")
    ax3.set_title("Fit quality for post-sweep wavelength estimation")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")
    quality_png_path = output_dir / f"{dat_path.stem}_{suffix}_fit_quality.png"
    fig3.savefig(quality_png_path, dpi=200)
    plt.close(fig3)

    csv_path = output_dir / f"{dat_path.stem}_{suffix}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("time_abs_s,time_rel_s,lambda_pm,drift_pm,fit_corr\n")
        for t_abs, t_rel, lam, drift, corr in zip(
            merged_time_abs_s,
            merged_time_rel_s,
            merged_lambda_pm,
            merged_drift_pm,
            merged_fit_corr,
        ):
            fout.write(f"{t_abs:.10f},{t_rel:.10f},{lam:.10f},{drift:.10f},{corr:.10f}\n")

    mat_path, script_path = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag=suffix,
        payload={
            "fiber_distance_m": fiber_distance_m[:, None],
            "fit_lag_indices": fit_lag_indices[:, None],
            "selected_sweep_index": np.array([[selected_sweep_index]], dtype=np.int32),
            "last_sweep_end_s": np.array([[last_sweep_end_s]], dtype=np.float64),
            "dominant_period_s": np.array([[dominant_period_s]], dtype=np.float64),
            "reset_period_override_ms": np.array(
                [[np.nan if args.reset_period_override_ms is None else float(args.reset_period_override_ms)]],
                dtype=np.float64,
            ),
            "reset_anchor_time_s": np.array(
                [[np.nan if args.reset_anchor_time_s is None else float(args.reset_anchor_time_s)]],
                dtype=np.float64,
            ),
            "baseline_window_start_m": np.array([[baseline_start_m]], dtype=np.float64),
            "baseline_window_end_m": np.array([[baseline_end_m]], dtype=np.float64),
            "even_time_rel_s": parity_results["even"]["time_rel_s"][:, None],
            "odd_time_rel_s": parity_results["odd"]["time_rel_s"][:, None],
            "even_lambda_pm": parity_results["even"]["lambda_pm"][:, None],
            "odd_lambda_pm": parity_results["odd"]["lambda_pm"][:, None],
            "even_lambda_pm_raw": parity_results["even"]["lambda_pm_raw"][:, None],
            "odd_lambda_pm_raw": parity_results["odd"]["lambda_pm_raw"][:, None],
            "even_drift_pm": parity_results["even"]["drift_pm"][:, None],
            "odd_drift_pm": parity_results["odd"]["drift_pm"][:, None],
            "even_fit_corr": parity_results["even"]["fit_corr"][:, None],
            "odd_fit_corr": parity_results["odd"]["fit_corr"][:, None],
            "merged_time_rel_s": merged_time_rel_s[:, None],
            "merged_lambda_pm": merged_lambda_pm[:, None],
            "merged_drift_pm": merged_drift_pm[:, None],
            "merged_fit_corr": merged_fit_corr[:, None],
            "merged_lambda_pm_rolling": merged_lambda_pm_rolling[:, None],
            "merged_drift_pm_rolling": merged_drift_pm_rolling[:, None],
            "merged_fit_corr_rolling": merged_fit_corr_rolling[:, None],
        },
    )

    print(f"file: {dat_path}")
    print(f"model_mat: {model_mat_path}")
    print(f"selected_sweep_index: {selected_sweep_index}")
    print(f"last_sweep_end_s: {last_sweep_end_s:.10f}")
    print(f"post_duration_s: {args.post_duration_s}")
    print(f"dominant_period_s: {dominant_period_s:.10f}")
    if args.reset_period_override_ms is not None:
        print(f"reset_period_override_ms: {args.reset_period_override_ms}")
    if args.reset_anchor_time_s is not None:
        print(f"reset_anchor_time_s: {args.reset_anchor_time_s}")
    print(f"fit_lag_count: {fit_lag_indices.size}")
    print(f"lambda_grid_min_pm: {args.lambda_grid_min_pm}")
    print(f"lambda_grid_max_pm: {args.lambda_grid_max_pm}")
    print(f"lambda_grid_step_pm: {args.lambda_grid_step_pm}")
    print(f"tracking_step_sigma_pm: {args.tracking_step_sigma_pm}")
    print(f"tracking_initial_window_traces: {args.tracking_initial_window_traces}")
    print(f"merged_trace_count: {merged_time_rel_s.size}")
    print(f"lambda_pm_start: {merged_lambda_pm[0]:.10f}")
    print(f"lambda_pm_end: {merged_lambda_pm[-1]:.10f}")
    print(f"drift_pm_end: {merged_drift_pm[-1]:.10f}")
    print(f"drift_pm_min: {np.min(merged_drift_pm):.10f}")
    print(f"drift_pm_max: {np.max(merged_drift_pm):.10f}")
    print(f"fit_corr_mean: {np.mean(merged_fit_corr):.10f}")
    print(f"fit_corr_min: {np.min(merged_fit_corr):.10f}")
    print(f"lambda_png_saved_to: {lambda_png_path}")
    print(f"drift_png_saved_to: {drift_png_path}")
    print(f"fit_quality_png_saved_to: {quality_png_path}")
    print(f"csv_saved_to: {csv_path}")
    print(f"matlab_data_saved_to: {mat_path}")
    print(f"matlab_script_saved_to: {script_path}")


if __name__ == "__main__":
    main()
