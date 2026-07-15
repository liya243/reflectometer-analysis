import argparse
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.signal import find_peaks

from raw_data import read_reflectograms, sellmeier_n


LIGHT_SPEED_M_PER_S = 299792458.0


def distance_axis_from_sampling_rate(n_samples, sampling_rate_hz, wavelength_um=1.55):
    n_eff = float(sellmeier_n(wavelength_um))
    distance_step_m = LIGHT_SPEED_M_PER_S / (2.0 * n_eff * float(sampling_rate_hz))
    return np.arange(n_samples, dtype=np.float64) * distance_step_m, distance_step_m


def moving_average_ignore_nan(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()

    kernel = np.ones(int(window), dtype=np.float64)
    filled = np.nan_to_num(values, nan=0.0)
    valid = np.isfinite(values).astype(np.float64)

    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")

    averaged = np.full(values.shape, np.nan, dtype=np.float64)
    mask = counts > 0.0
    averaged[mask] = sums[mask] / counts[mask]
    return averaged


def neighbor_correlation(trace_matrix):
    data = np.asarray(trace_matrix, dtype=np.float64)
    centered = data - data.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1, keepdims=True))

    normalized = np.full(data.shape, np.nan, dtype=np.float64)
    valid = rms[:, 0] > 0.0
    normalized[valid] = centered[valid] / rms[valid]
    return np.nanmean(normalized[:-1] * normalized[1:], axis=1)


def save_figure_bundle(fig, png_path):
    png_path = Path(png_path)
    svg_path = png_path.with_suffix(".svg")
    pdf_path = png_path.with_suffix(".pdf")
    pickle_path = png_path.with_suffix(".pickle")

    fig.savefig(png_path, dpi=200)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    with pickle_path.open("wb") as fout:
        pickle.dump(fig, fout)

    return {
        "png": png_path,
        "svg": svg_path,
        "pdf": pdf_path,
        "pickle": pickle_path,
    }


def matlab_safe_stem(text):
    safe = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not safe or not safe[0].isalpha():
        safe = f"fig_{safe}"
    return safe


def save_matlab_bundle(
    output_dir,
    stem,
    suffix_tag,
    time_axis_s,
    correlation,
    correlation_rolling,
    reset_times_s,
    reset_corr,
    reset_prominence,
    ignore_first_meters,
    rolling_window,
):
    output_dir = Path(output_dir)
    data_mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"

    savemat(
        data_mat_path,
        {
            "time_s": np.asarray(time_axis_s, dtype=np.float64),
            "neighbor_correlation": np.asarray(correlation, dtype=np.float64),
            "neighbor_correlation_rolling": np.asarray(correlation_rolling, dtype=np.float64),
            "reset_times_s": np.asarray(reset_times_s, dtype=np.float64),
            "reset_corr": np.asarray(reset_corr, dtype=np.float64),
            "reset_prominence": np.asarray(reset_prominence, dtype=np.float64),
            "ignore_first_meters": np.array([[ignore_first_meters]], dtype=np.float64),
            "rolling_window": np.array([[rolling_window]], dtype=np.int32),
        },
    )

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{data_mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Neighbor correlation vs time');
plot(data.time_s, data.neighbor_correlation, 'Color', [0.72 0.80 0.92], 'LineWidth', 0.5);
hold on;
plot(data.time_s, data.neighbor_correlation_rolling, 'k', 'LineWidth', 1.6);
scatter(data.reset_times_s, data.reset_corr, 30, 'r', 'filled');
grid on;
xlabel('Time (s)');
ylabel('Neighbor correlation');
title(sprintf('Корреляция соседних трасс во времени, первые %.1f m игнорируются', data.ignore_first_meters));
legend('Raw', sprintf('Rolling mean (%d)', data.rolling_window), 'Candidate resets', 'Location', 'best');

% Optional: save native MATLAB figure file after opening it.
% savefig(f1, fullfile(this_dir, '{stem}_{suffix_tag}.fig'));
"""
    script_path.write_text(script_text, encoding="utf-8")

    return {
        "mat": data_mat_path,
        "script": script_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Отследить корреляцию между соседними рефлектограммами во времени."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Каталог для выходных файлов",
    )
    parser.add_argument(
        "--scan-rate",
        type=float,
        default=None,
        help="Необязательная частота записи рефлектограмм в Hz",
    )
    parser.add_argument(
        "--ignore-first-meters",
        type=float,
        default=100.0,
        help="Игнорировать столько метров от начала каждой рефлектограммы",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=64,
        help="Окно скользящего среднего в трассах",
    )
    parser.add_argument(
        "--reset-min-distance-s",
        type=float,
        default=0.5,
        help="Минимальное временное расстояние между кандидатными моментами сброса",
    )
    parser.add_argument(
        "--reset-prominence",
        type=float,
        default=0.005,
        help="Минимальная выраженность провалов на сглаженной кривой соседней корреляции",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Число кандидатных моментов сброса для сохранения в summary CSV",
    )
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float64)

    distance_axis_m, distance_step_m = distance_axis_from_sampling_rate(
        result["real_segment_size"],
        result["sampling_rate"],
    )
    keep_mask = distance_axis_m >= float(args.ignore_first_meters)
    if not np.any(keep_mask):
        raise ValueError("ignore_first_meters removes the full reflectogram")
    cropped = data[:, keep_mask]

    corr = neighbor_correlation(cropped)
    time_axis_s = (np.arange(corr.size, dtype=np.float64) + 0.5) / result["scan_rate"]
    corr_rolling = moving_average_ignore_nan(corr, args.rolling_window)

    min_distance_traces = max(1, int(round(args.reset_min_distance_s * result["scan_rate"])))
    peaks, props = find_peaks(
        -corr_rolling,
        prominence=float(args.reset_prominence),
        distance=min_distance_traces,
    )

    order = np.argsort(corr_rolling[peaks])
    peaks = peaks[order]
    prominences = props["prominences"][order]
    if args.top_k > 0:
        peaks = peaks[: args.top_k]
        prominences = prominences[: args.top_k]

    reset_times_s = time_axis_s[peaks]
    reset_corr = corr[peaks]
    reset_corr_rolling = corr_rolling[peaks]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(time_axis_s, corr, color="#4C78A8", linewidth=0.35, alpha=0.45)
    ax.plot(time_axis_s, corr_rolling, color="#111111", linewidth=1.6, label=f"Rolling mean ({args.rolling_window})")
    if peaks.size > 0:
        ax.scatter(reset_times_s, reset_corr_rolling, s=28, color="#D62728", label="Кандидатные сбросы", zorder=5)
    ax.set_xlabel("Время (s)")
    ax.set_ylabel("Корреляция соседних трасс")
    ax.set_title("Корреляция между соседними трассами во времени")
    ax.grid(alpha=0.25)
    ax.legend()

    suffix_tag = f"neighbor_corr_ignore_first_{int(round(args.ignore_first_meters))}m"
    png_path = output_dir / f"{dat_path.stem}_{suffix_tag}.png"
    saved_paths = save_figure_bundle(fig, png_path)
    plt.close(fig)

    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag=suffix_tag,
        time_axis_s=time_axis_s,
        correlation=corr,
        correlation_rolling=corr_rolling,
        reset_times_s=reset_times_s,
        reset_corr=reset_corr_rolling,
        reset_prominence=prominences,
        ignore_first_meters=float(args.ignore_first_meters),
        rolling_window=args.rolling_window,
    )

    csv_path = output_dir / f"{dat_path.stem}_{suffix_tag}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("left_index,right_index,time_s,neighbor_correlation,neighbor_correlation_rolling\n")
        for idx, (time_s, c_raw, c_roll) in enumerate(zip(time_axis_s, corr, corr_rolling)):
            fout.write(f"{idx},{idx+1},{time_s:.10f},{c_raw:.10f},{c_roll:.10f}\n")

    reset_csv_path = output_dir / f"{dat_path.stem}_{suffix_tag}_candidate_resets.csv"
    with reset_csv_path.open("w", encoding="utf-8") as fout:
        fout.write("rank,left_index,right_index,time_s,neighbor_correlation,neighbor_correlation_rolling,prominence\n")
        for rank, (idx, time_s, c_raw, c_roll, prom) in enumerate(
            zip(peaks, reset_times_s, reset_corr, reset_corr_rolling, prominences),
            start=1,
        ):
            fout.write(f"{rank},{idx},{idx+1},{time_s:.10f},{c_raw:.10f},{c_roll:.10f},{prom:.10f}\n")

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"duration_s: {result['refls_count'] / result['scan_rate']:.6f}")
    print(f"distance_step_m: {distance_step_m:.6f}")
    print(f"ignore_first_meters: {args.ignore_first_meters}")
    print(f"kept_samples: {cropped.shape[1]}")
    print(f"kept_distance_start_m: {distance_axis_m[keep_mask][0]:.6f}")
    print(f"kept_distance_end_m: {distance_axis_m[keep_mask][-1]:.6f}")
    print(f"neighbor_corr_mean: {np.mean(corr):.6f}")
    print(f"neighbor_corr_std: {np.std(corr):.6f}")
    print(f"neighbor_corr_min: {np.min(corr):.6f} at left index {int(np.argmin(corr))}")
    print(f"candidate_reset_count: {len(peaks)}")
    if len(peaks) > 0:
        print(f"deepest_candidate_time_s: {reset_times_s[0]:.6f}")
        print(f"deepest_candidate_neighbor_corr: {reset_corr[0]:.6f}")
    print(f"csv_saved_to: {csv_path}")
    print(f"candidate_resets_csv_saved_to: {reset_csv_path}")
    print(f"plot_png_saved_to: {saved_paths['png']}")
    print(f"plot_svg_saved_to: {saved_paths['svg']}")
    print(f"plot_pdf_saved_to: {saved_paths['pdf']}")
    print(f"plot_pickle_saved_to: {saved_paths['pickle']}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
