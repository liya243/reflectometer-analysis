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


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def dominant_score(trace_matrix):
    x = np.asarray(trace_matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = x.T @ x
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_vector = eigvecs[:, np.argmax(eigvals)]
    score = x @ principal_vector
    return score


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


def estimate_period_from_fft(score, scan_rate_hz, min_period_s, max_period_s):
    score = np.asarray(score, dtype=np.float64)
    if score.size < 4:
        raise ValueError("Need at least 4 traces to estimate period")

    centered = score - np.mean(score)
    window = np.hanning(score.size)
    freqs_hz = np.fft.rfftfreq(score.size, d=1.0 / float(scan_rate_hz))
    power = np.abs(np.fft.rfft(centered * window)) ** 2

    valid = freqs_hz > 0.0
    if min_period_s is not None:
        valid &= freqs_hz <= (1.0 / float(min_period_s))
    if max_period_s is not None:
        valid &= freqs_hz >= (1.0 / float(max_period_s))
    if not np.any(valid):
        raise ValueError("No FFT bins remain inside the requested period range")

    valid_indices = np.flatnonzero(valid)
    dominant_idx = valid_indices[np.argmax(power[valid])]
    dominant_freq_hz = float(freqs_hz[dominant_idx])
    dominant_period_s = 1.0 / dominant_freq_hz
    return dominant_period_s, dominant_freq_hz, freqs_hz, power, valid


def robust_threshold(values, sigma):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0.0:
        robust_sigma = float(np.std(values))
    return median + float(sigma) * robust_sigma


def detect_period_locked_spikes(abs_signal, period_traces, amplitude_threshold, phase_half_width_traces):
    abs_signal = np.asarray(abs_signal, dtype=np.float64)
    if abs_signal.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0,
        )

    if period_traces < 2:
        raise ValueError("period_traces must be at least 2")

    phase_sum = np.zeros(period_traces, dtype=np.float64)
    phase_count = np.zeros(period_traces, dtype=np.float64)
    phase_idx = np.arange(abs_signal.size, dtype=np.int64) % int(period_traces)
    np.add.at(phase_sum, phase_idx, abs_signal)
    np.add.at(phase_count, phase_idx, 1.0)
    phase_profile = phase_sum / np.maximum(phase_count, 1.0)
    dominant_phase = int(np.argmax(phase_profile))

    peaks = []
    strengths = []
    center = dominant_phase
    while center < abs_signal.size:
        left = max(0, center - phase_half_width_traces)
        right = min(abs_signal.size, center + phase_half_width_traces + 1)
        if right - left >= 3:
            local = abs_signal[left:right]
            local_idx = left + int(np.argmax(local))
            local_strength = float(abs_signal[local_idx])
            if local_strength >= amplitude_threshold:
                peaks.append(local_idx)
                strengths.append(local_strength)
        center += int(period_traces)

    return (
        np.asarray(peaks, dtype=np.int64),
        np.asarray(strengths, dtype=np.float64),
        phase_profile,
        dominant_phase,
    )


def frame_change_metric(trace_matrix):
    x = np.asarray(trace_matrix, dtype=np.float64)
    if x.shape[0] < 2:
        return np.array([], dtype=np.float64)
    return np.sqrt(np.mean(np.diff(x, axis=0) ** 2, axis=1))


def refine_peaks_with_metric(coarse_peaks, metric, search_radius_traces):
    metric = np.asarray(metric, dtype=np.float64)
    refined = []
    strengths = []
    for peak in np.asarray(coarse_peaks, dtype=np.int64):
        left = max(0, int(peak) - int(search_radius_traces))
        right = min(metric.size, int(peak) + int(search_radius_traces) + 1)
        if right <= left:
            continue
        local_idx = left + int(np.argmax(metric[left:right]))
        refined.append(local_idx)
        strengths.append(float(metric[local_idx]))

    if len(refined) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    refined = np.asarray(refined, dtype=np.int64)
    strengths = np.asarray(strengths, dtype=np.float64)

    keep = np.ones(refined.size, dtype=bool)
    for idx in range(1, refined.size):
        if refined[idx] == refined[idx - 1]:
            if strengths[idx] > strengths[idx - 1]:
                keep[idx - 1] = False
            else:
                keep[idx] = False

    return refined[keep], strengths[keep]


def main():
    parser = argparse.ArgumentParser(
        description="Detect saw-reset moments from the dominant global modulation mode."
    )
    parser.add_argument("dat_path", help="Path to the .dat file")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory for output files",
    )
    parser.add_argument(
        "--ignore-first-meters",
        type=float,
        default=100.0,
        help="Ignore this many meters from the beginning of each reflectogram",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=64,
        help="Smoothing window in traces",
    )
    parser.add_argument(
        "--min-period-s",
        type=float,
        default=0.05,
        help="Minimum period to consider in the FFT search",
    )
    parser.add_argument(
        "--max-period-s",
        type=float,
        default=None,
        help="Maximum period to consider in the FFT search; default is half the record duration",
    )
    parser.add_argument(
        "--prominence-sigma",
        type=float,
        default=3.0,
        help="Detection threshold in robust sigma units for |d(score)/dt|",
    )
    parser.add_argument(
        "--parity",
        choices=["all", "even", "odd"],
        default="all",
        help="Use all reflectograms or only one parity subset",
    )
    parser.add_argument(
        "--refine-window-fraction",
        type=float,
        default=0.15,
        help="Refine each coarse reset time within this fraction of the detected period using frame-to-frame change",
    )
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path))
    data = np.asarray(result["data"], dtype=np.float64)
    distance_axis_m, distance_step_m = distance_axis_from_sampling_rate(
        result["real_segment_size"],
        result["sampling_rate"],
    )
    keep_mask = distance_axis_m >= float(args.ignore_first_meters)
    if not np.any(keep_mask):
        raise ValueError("ignore_first_meters removes the full reflectogram")
    cropped = data[:, keep_mask]
    cropped, selected_global_indices = select_parity_subset(cropped, args.parity)
    if cropped.shape[0] < 4:
        raise ValueError("Not enough traces remain after parity selection")
    global_step = float(np.median(np.diff(selected_global_indices))) if selected_global_indices.size > 1 else 1.0
    effective_scan_rate_hz = float(result["scan_rate"]) / global_step

    score = dominant_score(cropped)
    if np.mean(np.diff(score[: min(5000, score.size - 1)])) < 0:
        score = -score

    score_smooth = moving_average(score, args.rolling_window)
    score_derivative = np.diff(score_smooth)
    abs_derivative = np.abs(score_derivative)
    derivative_time_s = (selected_global_indices[:-1] + selected_global_indices[1:]) / (2.0 * result["scan_rate"])

    duration_s = (selected_global_indices[-1] - selected_global_indices[0]) / result["scan_rate"]
    max_period_s = args.max_period_s if args.max_period_s is not None else 0.5 * duration_s
    dominant_period_s, dominant_freq_hz, freqs_hz, power, valid_bins = estimate_period_from_fft(
        score_smooth,
        effective_scan_rate_hz,
        min_period_s=args.min_period_s,
        max_period_s=max_period_s,
    )
    period_traces = max(2, int(round(dominant_period_s * effective_scan_rate_hz)))
    phase_half_width_traces = max(1, int(round(0.3 * period_traces)))
    amplitude_threshold = robust_threshold(abs_derivative, args.prominence_sigma)
    coarse_peaks, coarse_strength, phase_profile, dominant_phase = detect_period_locked_spikes(
        abs_derivative,
        period_traces=period_traces,
        amplitude_threshold=amplitude_threshold,
        phase_half_width_traces=phase_half_width_traces,
    )
    change_metric = frame_change_metric(cropped)
    refine_window_traces = max(1, int(round(float(args.refine_window_fraction) * period_traces)))
    peaks, reset_strength = refine_peaks_with_metric(
        coarse_peaks,
        change_metric,
        search_radius_traces=refine_window_traces,
    )

    reset_times_s = derivative_time_s[peaks] if peaks.size > 0 else np.array([], dtype=np.float64)
    reset_signed_derivative = score_derivative[peaks] if peaks.size > 0 else np.array([], dtype=np.float64)
    if peaks.size > 1:
        period_estimates_s = np.diff(reset_times_s)
        period_median_s = float(np.median(period_estimates_s))
        period_mean_s = float(np.mean(period_estimates_s))
    else:
        period_estimates_s = np.array([], dtype=np.float64)
        period_median_s = float("nan")
        period_mean_s = float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False, constrained_layout=True)

    time_axis_s = selected_global_indices.astype(np.float64) / result["scan_rate"]
    axes[0].plot(time_axis_s, score, color="#4C78A8", linewidth=0.3, alpha=0.35)
    axes[0].plot(time_axis_s, score_smooth, color="#111111", linewidth=1.5, label=f"Rolling mean ({args.rolling_window})")
    if peaks.size > 0:
        axes[0].scatter(reset_times_s, score_smooth[peaks], color="#D62728", s=28, label="Detected resets", zorder=5)
    axes[0].set_ylabel("Dominant score")
    axes[0].set_title(
        f"Reset detection from dominant global modulation mode (FFT period {dominant_period_s:.4f} s)"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    period_axis_s = 1.0 / freqs_hz[valid_bins]
    order = np.argsort(period_axis_s)
    axes[1].plot(period_axis_s[order], power[valid_bins][order], color="#4C78A8", linewidth=1.0)
    axes[1].axvline(dominant_period_s, color="#D62728", linestyle="--", linewidth=1.0, label="Detected period")
    axes[1].set_xlabel("Period (s)")
    axes[1].set_ylabel("FFT power")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(derivative_time_s, change_metric, color="#2E86AB", linewidth=0.5, label="Frame-to-frame RMS change")
    if peaks.size > 0:
        axes[2].scatter(reset_times_s, change_metric[peaks], color="#D62728", s=28, zorder=5, label="Detected resets")
    for k in range(0, int(np.ceil((abs_derivative.size - dominant_phase) / float(period_traces)))):
        local_center = dominant_phase + k * period_traces
        if local_center + 1 >= selected_global_indices.size:
            break
        center_time_s = (selected_global_indices[local_center] + selected_global_indices[local_center + 1]) / (
            2.0 * result["scan_rate"]
        )
        if center_time_s <= derivative_time_s[-1]:
            axes[2].axvline(center_time_s, color="#999999", linewidth=0.4, alpha=0.25)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Trace change")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    suffix_tag = f"_{args.parity}" if args.parity != "all" else ""
    suffix_tag += f"_ignore_first_{int(round(args.ignore_first_meters))}m"
    png_path = output_dir / f"{dat_path.stem}_reset_detection{suffix_tag}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    reset_csv_path = output_dir / f"{dat_path.stem}_reset_detection{suffix_tag}.csv"
    with reset_csv_path.open("w", encoding="utf-8") as fout:
        fout.write("rank,time_s,left_index,right_index,abs_derivative_strength,signed_derivative,cycle_period_s\n")
        for rank, (peak, t, abs_strength, signed_strength) in enumerate(
            zip(peaks, reset_times_s, reset_strength, reset_signed_derivative),
            start=1,
        ):
            fout.write(
                f"{rank},{t:.10f},{peak},{peak+1},{abs_strength:.10f},{signed_strength:.10f},{dominant_period_s:.10f}\n"
            )

    print(f"file: {dat_path}")
    print(f"parity: {args.parity}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"effective_scan_rate_hz: {effective_scan_rate_hz:.6f}")
    print(f"duration_s: {duration_s:.6f}")
    print(f"distance_step_m: {distance_step_m:.6f}")
    print(f"ignore_first_meters: {args.ignore_first_meters}")
    print(f"kept_distance_start_m: {distance_axis_m[keep_mask][0]:.6f}")
    print(f"kept_distance_end_m: {distance_axis_m[keep_mask][-1]:.6f}")
    print(f"rolling_window: {args.rolling_window}")
    print(f"fft_min_period_s: {args.min_period_s}")
    print(f"fft_max_period_s: {max_period_s}")
    print(f"dominant_period_s: {dominant_period_s:.10f}")
    print(f"dominant_frequency_hz: {dominant_freq_hz:.10f}")
    print(f"period_traces: {period_traces}")
    print(f"dominant_phase_trace: {dominant_phase}")
    print(f"phase_half_width_traces: {phase_half_width_traces}")
    print(f"refine_window_traces: {refine_window_traces}")
    print(f"amplitude_threshold: {amplitude_threshold:.10f}")
    print(f"reset_count: {peaks.size}")
    print(f"reset_times_s: {np.array2string(reset_times_s, precision=6, separator=', ')}")
    if peaks.size > 1:
        print(f"period_estimates_s: {np.array2string(period_estimates_s, precision=6, separator=', ')}")
        print(f"median_period_s: {period_median_s:.6f}")
        print(f"mean_period_s: {period_mean_s:.6f}")
    print(f"plot_png_saved_to: {png_path}")
    print(f"reset_csv_saved_to: {reset_csv_path}")


if __name__ == "__main__":
    main()
