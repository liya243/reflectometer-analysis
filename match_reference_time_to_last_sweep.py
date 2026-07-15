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
    out = np.full_like(centered, np.nan, dtype=np.float64)
    valid = rms[:, 0] > 0.0
    out[valid] = centered[valid] / rms[valid]
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


def parabolic_peak_refine(x, y, best_idx):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    best_idx = int(best_idx)
    if best_idx <= 0 or best_idx >= x.size - 1:
        return float(x[best_idx]), float(y[best_idx])
    xs = x[best_idx - 1 : best_idx + 2]
    ys = y[best_idx - 1 : best_idx + 2]
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        return float(x[best_idx]), float(y[best_idx])
    coeff = np.polyfit(xs, ys, deg=2)
    if coeff[0] >= 0.0:
        return float(x[best_idx]), float(y[best_idx])
    peak_x = -coeff[1] / (2.0 * coeff[0])
    if peak_x < xs[0] or peak_x > xs[-1]:
        return float(x[best_idx]), float(y[best_idx])
    peak_y = np.polyval(coeff, peak_x)
    return float(peak_x), float(peak_y)


def fit_correlation_for_mask(
    parity_data,
    parity_time_s,
    coord_mask,
    sweep_start_s,
    sweep_end_s,
    reference_time_s,
    sweep_span_pm,
    reference_half_window_traces,
    sweep_half_window_traces,
):
    sweep_mask = (parity_time_s >= float(sweep_start_s)) & (parity_time_s < float(sweep_end_s))
    if np.count_nonzero(sweep_mask) < 3:
        raise ValueError("Selected sweep has too few traces")

    target_idx = int(np.argmin(np.abs(parity_time_s - float(reference_time_s))))
    target_time_s = float(parity_time_s[target_idx])
    ref_start = max(0, target_idx - int(reference_half_window_traces))
    ref_stop = min(parity_data.shape[0], target_idx + int(reference_half_window_traces) + 1)

    sweep_source = parity_data[sweep_mask][:, coord_mask]
    sweep_time_s = parity_time_s[sweep_mask]
    sweep = np.empty_like(sweep_source)
    for row in range(sweep_source.shape[0]):
        row_start = max(0, row - int(sweep_half_window_traces))
        row_stop = min(sweep_source.shape[0], row + int(sweep_half_window_traces) + 1)
        sweep[row] = np.mean(sweep_source[row_start:row_stop], axis=0)
    reference = np.mean(parity_data[ref_start:ref_stop, coord_mask], axis=0, keepdims=True)

    sweep_norm = center_and_rms_normalize_rows(sweep)
    reference_norm = center_and_rms_normalize_rows(reference)[0]
    corr = sweep_norm @ reference_norm / float(np.count_nonzero(coord_mask))
    lambda_pm = float(sweep_span_pm) * (sweep_time_s - float(sweep_start_s)) / (float(sweep_end_s) - float(sweep_start_s))

    best_idx = int(np.nanargmax(corr))
    refined_lambda_pm, refined_corr = parabolic_peak_refine(lambda_pm, corr, best_idx)
    refined_time_s = float(sweep_start_s) + refined_lambda_pm / float(sweep_span_pm) * (float(sweep_end_s) - float(sweep_start_s))

    return {
        "target_idx": target_idx,
        "target_time_s": target_time_s,
        "reference_average_start_time_s": float(parity_time_s[ref_start]),
        "reference_average_end_time_s": float(parity_time_s[ref_stop - 1]),
        "reference_average_count": int(ref_stop - ref_start),
        "sweep_average_count_max": int(2 * int(sweep_half_window_traces) + 1),
        "sweep_time_s": sweep_time_s,
        "lambda_pm": lambda_pm,
        "corr": corr,
        "best_idx": best_idx,
        "best_time_s": float(sweep_time_s[best_idx]),
        "best_lambda_pm": float(lambda_pm[best_idx]),
        "best_corr": float(corr[best_idx]),
        "refined_time_s": refined_time_s,
        "refined_lambda_pm": refined_lambda_pm,
        "refined_corr": refined_corr,
        "target_trace": parity_data[target_idx, coord_mask],
        "best_sweep_trace": parity_data[np.flatnonzero(sweep_mask)[best_idx], coord_mask],
    }


def save_matlab_bundle(output_dir, stem, suffix, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix}_matlab_data.mat"
    script_path = output_dir / f"{matlab_safe_stem(f'open_{stem}_{suffix}_in_matlab')}.m"
    savemat(mat_path, payload)
    script_path.write_text(
        f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

figure('Color', 'w', 'Name', 'Reference-to-last-sweep correlation');
subplot(2, 1, 1);
plot(data.even_lambda_pm, data.even_corr, 'LineWidth', 1.2);
hold on; xline(data.even_best_lambda_pm, 'r--', 'LineWidth', 1.2);
grid on; xlabel('\\Delta\\lambda inside last sweep (pm)'); ylabel('Correlation');
title(sprintf('Even, best %.4f pm, corr %.4f', data.even_best_lambda_pm, data.even_best_corr));

subplot(2, 1, 2);
plot(data.odd_lambda_pm, data.odd_corr, 'LineWidth', 1.2);
hold on; xline(data.odd_best_lambda_pm, 'r--', 'LineWidth', 1.2);
grid on; xlabel('\\Delta\\lambda inside last sweep (pm)'); ylabel('Correlation');
title(sprintf('Odd, best %.4f pm, corr %.4f', data.odd_best_lambda_pm, data.odd_best_corr));

figure('Color', 'w', 'Name', 'Matched traces');
subplot(2, 1, 1);
plot(data.coord_m, data.even_target_trace, 'k', 'LineWidth', 1.0);
hold on; plot(data.coord_m, data.even_best_sweep_trace, 'r', 'LineWidth', 1.0);
grid on; xlabel('Coordinate (m)'); ylabel('Signal');
title('Even: reference at requested time vs best last-sweep trace');
legend('Reference', 'Best last sweep', 'Location', 'best');

subplot(2, 1, 2);
plot(data.coord_m, data.odd_target_trace, 'k', 'LineWidth', 1.0);
hold on; plot(data.coord_m, data.odd_best_sweep_trace, 'r', 'LineWidth', 1.0);
grid on; xlabel('Coordinate (m)'); ylabel('Signal');
title('Odd: reference at requested time vs best last-sweep trace');
legend('Reference', 'Best last sweep', 'Location', 'best');
""",
        encoding="utf-8",
    )
    return mat_path, script_path


def main():
    parser = argparse.ArgumentParser(
        description="Match a post-modulation reference trace to the last wavelength sweep by direct correlation."
    )
    parser.add_argument("dat_path", help="Path to .dat file")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Directory for outputs")
    parser.add_argument("--scan-rate", type=float, default=None, help="Optional scan rate override")
    parser.add_argument("--fiber-z-min", type=float, default=110.0, help="Start of real fiber region in meters")
    parser.add_argument("--fiber-z-max", type=float, default=360.0, help="End of real fiber region in meters")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Subtract per-trace baseline from last this many meters")
    parser.add_argument("--reset-period-ms", type=float, default=76.8, help="Accepted sweep period in ms")
    parser.add_argument("--reset-anchor-time-s", type=float, default=0.0919, help="One accepted reset/sweep-boundary time in seconds")
    parser.add_argument("--max-reset-time-s", type=float, default=4.45, help="Last accepted reset-grid upper bound")
    parser.add_argument("--sweep-index", type=int, default=-1, help="Sweep interval index; -1 means last complete sweep")
    parser.add_argument("--sweep-span-pm", type=float, default=3.125, help="Wavelength span of one sweep")
    parser.add_argument("--reference-time-s", type=float, default=6.0, help="Requested reference time after modulation")
    parser.add_argument("--reference-half-window-traces", type=int, default=0, help="Average this many same-parity traces before/after the reference")
    parser.add_argument("--sweep-half-window-traces", type=int, default=0, help="Average this many same-parity last-sweep traces before/after each tested point")
    parser.add_argument("--exclude-z-min", type=float, default=None, help="Optional disturbed-zone start to exclude")
    parser.add_argument("--exclude-z-max", type=float, default=None, help="Optional disturbed-zone end to exclude")
    parser.add_argument("--suffix", default=None, help="Optional output suffix")
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float64)
    distance_axis_m, _, _ = distance_axis_from_sampling_rate(result["real_segment_size"], result["sampling_rate"])
    data, _, _, baseline_start_m, baseline_end_m = subtract_trace_baseline_from_tail(
        data,
        distance_axis_m,
        args.baseline_tail_m,
    )

    base_coord_mask = (distance_axis_m >= float(args.fiber_z_min)) & (distance_axis_m <= float(args.fiber_z_max))
    if args.exclude_z_min is not None and args.exclude_z_max is not None:
        exclude_mask = (distance_axis_m >= float(args.exclude_z_min)) & (distance_axis_m <= float(args.exclude_z_max))
        coord_mask = base_coord_mask & ~exclude_mask
        mask_tag = f"exclude_{args.exclude_z_min:g}_{args.exclude_z_max:g}m"
    else:
        coord_mask = base_coord_mask
        mask_tag = "full_fiber"
    if np.count_nonzero(coord_mask) < 4:
        raise ValueError("Correlation coordinate mask is too small")

    reset_times_s = build_periodic_reset_grid(
        anchor_time_s=args.reset_anchor_time_s,
        period_s=1e-3 * float(args.reset_period_ms),
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

    parity_results = {}
    for parity in ["even", "odd"]:
        parity_data, parity_global_indices = select_parity_subset(data, parity)
        parity_time_s = parity_global_indices.astype(np.float64) / float(result["scan_rate"])
        parity_results[parity] = fit_correlation_for_mask(
            parity_data,
            parity_time_s,
            coord_mask,
            sweep_start_s,
            sweep_end_s,
            args.reference_time_s,
            args.sweep_span_pm,
            args.reference_half_window_traces,
            args.sweep_half_window_traces,
        )

    suffix = args.suffix or f"reference_{args.reference_time_s:g}s_match_last_sweep_{mask_tag}"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, parity in enumerate(["even", "odd"]):
        res = parity_results[parity]
        axes[row, 0].plot(res["lambda_pm"], res["corr"], linewidth=1.25)
        axes[row, 0].axvline(res["best_lambda_pm"], color="#D62728", linestyle="--", linewidth=1.1, label="Best trace")
        axes[row, 0].axvline(res["refined_lambda_pm"], color="#111111", linestyle=":", linewidth=1.1, label="Parabolic peak")
        axes[row, 0].set_title(
            f"{parity}: best {res['best_lambda_pm']:.4f} pm, corr {res['best_corr']:.4f}"
        )
        axes[row, 0].set_xlabel("Delta lambda inside last sweep (pm)")
        axes[row, 0].set_ylabel("Correlation")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 0].legend(loc="best")

        axes[row, 1].plot(distance_axis_m[coord_mask], res["target_trace"], color="#111111", linewidth=0.9, label="6 s reference")
        axes[row, 1].plot(distance_axis_m[coord_mask], res["best_sweep_trace"], color="#D62728", linewidth=0.9, label="Best last sweep")
        axes[row, 1].set_title(f"{parity}: trace overlay")
        axes[row, 1].set_xlabel("Coordinate (m)")
        axes[row, 1].set_ylabel("Signal, baseline subtracted")
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].legend(loc="best")
    fig.suptitle(
        f"Reference at {args.reference_time_s:.4f} s matched to last sweep {sweep_start_s:.4f}-{sweep_end_s:.4f} s"
    )
    png_path = output_dir / f"{dat_path.stem}_{suffix}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    csv_path = output_dir / f"{dat_path.stem}_{suffix}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("parity,sweep_time_s,lambda_pm,corr\n")
        for parity in ["even", "odd"]:
            res = parity_results[parity]
            for t_s, lam_pm, corr in zip(res["sweep_time_s"], res["lambda_pm"], res["corr"]):
                fout.write(f"{parity},{t_s:.10f},{lam_pm:.10f},{corr:.10f}\n")

    payload = {
        "coord_m": distance_axis_m[coord_mask][:, None],
        "reference_time_requested_s": np.array([[args.reference_time_s]], dtype=np.float64),
        "sweep_start_s": np.array([[sweep_start_s]], dtype=np.float64),
        "sweep_end_s": np.array([[sweep_end_s]], dtype=np.float64),
        "sweep_span_pm": np.array([[args.sweep_span_pm]], dtype=np.float64),
        "baseline_window_start_m": np.array([[baseline_start_m]], dtype=np.float64),
        "baseline_window_end_m": np.array([[baseline_end_m]], dtype=np.float64),
    }
    for parity in ["even", "odd"]:
        res = parity_results[parity]
        prefix = f"{parity}_"
        payload.update(
            {
                f"{prefix}target_time_s": np.array([[res["target_time_s"]]], dtype=np.float64),
                f"{prefix}reference_average_start_time_s": np.array([[res["reference_average_start_time_s"]]], dtype=np.float64),
                f"{prefix}reference_average_end_time_s": np.array([[res["reference_average_end_time_s"]]], dtype=np.float64),
                f"{prefix}reference_average_count": np.array([[res["reference_average_count"]]], dtype=np.int32),
                f"{prefix}sweep_average_count_max": np.array([[res["sweep_average_count_max"]]], dtype=np.int32),
                f"{prefix}lambda_pm": res["lambda_pm"][:, None],
                f"{prefix}corr": res["corr"][:, None],
                f"{prefix}best_time_s": np.array([[res["best_time_s"]]], dtype=np.float64),
                f"{prefix}best_lambda_pm": np.array([[res["best_lambda_pm"]]], dtype=np.float64),
                f"{prefix}best_corr": np.array([[res["best_corr"]]], dtype=np.float64),
                f"{prefix}refined_time_s": np.array([[res["refined_time_s"]]], dtype=np.float64),
                f"{prefix}refined_lambda_pm": np.array([[res["refined_lambda_pm"]]], dtype=np.float64),
                f"{prefix}refined_corr": np.array([[res["refined_corr"]]], dtype=np.float64),
                f"{prefix}target_trace": res["target_trace"][:, None],
                f"{prefix}best_sweep_trace": res["best_sweep_trace"][:, None],
            }
        )
    mat_path, script_path = save_matlab_bundle(output_dir, dat_path.stem, suffix, payload)

    print(f"file: {dat_path}")
    print(f"reference_time_requested_s: {args.reference_time_s:.10f}")
    print(f"sweep_start_s: {sweep_start_s:.10f}")
    print(f"sweep_end_s: {sweep_end_s:.10f}")
    print(f"coord_window_m: {distance_axis_m[coord_mask][0]:.6f}..{distance_axis_m[coord_mask][-1]:.6f}")
    print(f"coord_count: {np.count_nonzero(coord_mask)}")
    for parity in ["even", "odd"]:
        res = parity_results[parity]
        print(f"{parity}_target_time_s: {res['target_time_s']:.10f}")
        print(f"{parity}_reference_average_window_s: {res['reference_average_start_time_s']:.10f}..{res['reference_average_end_time_s']:.10f}")
        print(f"{parity}_reference_average_count: {res['reference_average_count']}")
        print(f"{parity}_sweep_average_count_max: {res['sweep_average_count_max']}")
        print(f"{parity}_best_lambda_pm: {res['best_lambda_pm']:.10f}")
        print(f"{parity}_best_corr: {res['best_corr']:.10f}")
        print(f"{parity}_refined_lambda_pm: {res['refined_lambda_pm']:.10f}")
        print(f"{parity}_refined_corr: {res['refined_corr']:.10f}")
    print(f"png_saved_to: {png_path}")
    print(f"csv_saved_to: {csv_path}")
    print(f"matlab_data_saved_to: {mat_path}")
    print(f"matlab_script_saved_to: {script_path}")


if __name__ == "__main__":
    main()
