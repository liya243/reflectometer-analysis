import argparse
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from raw_data import read_reflectograms, sellmeier_n


LIGHT_SPEED_M_PER_S = 299792458.0


def center_and_rms_normalize(trace_matrix):
    data = np.asarray(trace_matrix, dtype=np.float64)
    centered = data - data.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1, keepdims=True))

    normalized = np.full(data.shape, np.nan, dtype=np.float64)
    valid = rms[:, 0] > 0.0
    normalized[valid] = centered[valid] / rms[valid]
    return normalized


def correlation_against_reference(trace_matrix, reference_index=0):
    normalized = center_and_rms_normalize(trace_matrix)
    reference = normalized[int(reference_index)]
    return np.nanmean(normalized * reference[None, :], axis=1)


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


def distance_axis_from_sampling_rate(n_samples, sampling_rate_hz, wavelength_um=1.55):
    n_eff = float(sellmeier_n(wavelength_um))
    distance_step_m = LIGHT_SPEED_M_PER_S / (2.0 * n_eff * float(sampling_rate_hz))
    return np.arange(n_samples, dtype=np.float64) * distance_step_m, distance_step_m, n_eff


def correlation_to_abs_delta_lambda_pm(correlation, coeff_pm_inv2):
    correlation = np.asarray(correlation, dtype=np.float64)
    argument = np.maximum(0.0, 1.0 - correlation) / float(coeff_pm_inv2)
    return np.sqrt(argument)


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


def fit_linear_trend(values):
    indices = np.arange(values.size, dtype=np.float64)
    finite = np.isfinite(values)
    slope, intercept = np.polyfit(indices[finite], values[finite], 1)
    total_change = slope * (indices[finite][-1] - indices[finite][0])
    return slope, intercept, total_change


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
    ref_tag,
    time_axis_s,
    correlation,
    correlation_rolling,
    abs_delta_lambda_pm,
    abs_delta_lambda_rolling,
    reference_index,
    scan_rate_hz,
    duration_s,
    coeff_pm_inv2,
    rolling_window,
):
    output_dir = Path(output_dir)
    data_mat_path = output_dir / f"{stem}_{ref_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{ref_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"

    savemat(
        data_mat_path,
        {
            "time_s": np.asarray(time_axis_s, dtype=np.float64),
            "correlation": np.asarray(correlation, dtype=np.float64),
            "correlation_rolling": np.asarray(correlation_rolling, dtype=np.float64),
            "abs_delta_lambda_pm": np.asarray(abs_delta_lambda_pm, dtype=np.float64),
            "abs_delta_lambda_rolling": np.asarray(abs_delta_lambda_rolling, dtype=np.float64),
            "reference_index": np.array([[reference_index]], dtype=np.int32),
            "scan_rate_hz": np.array([[scan_rate_hz]], dtype=np.float64),
            "duration_s": np.array([[duration_s]], dtype=np.float64),
            "corr_to_dlambda_coeff_pm_inv2": np.array([[coeff_pm_inv2]], dtype=np.float64),
            "rolling_window": np.array([[rolling_window]], dtype=np.int32),
        },
    )

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{data_mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Correlation vs time');
plot(data.time_s, data.correlation, 'Color', [0.72 0.80 0.92], 'LineWidth', 0.5);
hold on;
plot(data.time_s, data.correlation_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Time (s)');
ylabel('Correlation');
title(sprintf('Корреляция во времени, референсная рефлектограмма %d', data.reference_index));
legend(sprintf('Корреляция'), sprintf('Скользящее среднее (%d)', data.rolling_window), 'Location', 'best');

f2 = figure('Color', 'w', 'Name', 'Absolute wavelength shift vs time');
plot(data.time_s, data.abs_delta_lambda_pm, 'Color', [1.00 0.82 0.62], 'LineWidth', 0.5);
hold on;
plot(data.time_s, data.abs_delta_lambda_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Time (s)');
ylabel('|\\Delta\\lambda| (pm)');
title(sprintf('Абсолютный сдвиг длины волны во времени, референсная рефлектограмма %d', data.reference_index));
legend(sprintf('|\\Delta\\lambda|'), sprintf('Скользящее среднее (%d)', data.rolling_window), 'Location', 'best');

% Optional: save native MATLAB figure files after opening them.
% savefig(f1, fullfile(this_dir, '{stem}_corr_vs_{ref_tag}.fig'));
% savefig(f2, fullfile(this_dir, '{stem}_abs_delta_lambda_pm_vs_{ref_tag}.fig'));
"""
    script_path.write_text(script_text, encoding="utf-8")

    return {
        "mat": data_mat_path,
        "script": script_path,
    }


def write_csv(path, time_axis_s, correlation, abs_delta_lambda_pm, global_indices):
    local_indices = np.arange(correlation.size, dtype=np.int64)

    with path.open("w", encoding="utf-8") as fout:
        fout.write("local_index,global_index,time_s,correlation,abs_delta_lambda_pm\n")
        for local_idx, global_idx, time_s, corr, abs_dl_pm in zip(
            local_indices,
            global_indices,
            time_axis_s,
            correlation,
            abs_delta_lambda_pm,
        ):
            time_text = "nan" if not np.isfinite(time_s) else f"{time_s:.10f}"
            corr_text = "nan" if not np.isfinite(corr) else f"{corr:.10f}"
            abs_dl_text = "nan" if not np.isfinite(abs_dl_pm) else f"{abs_dl_pm:.10f}"
            fout.write(f"{local_idx},{global_idx},{time_text},{corr_text},{abs_dl_text}\n")


def make_plot(path, time_axis_s, correlation, rolling_window, reference_index):
    rolling = moving_average_ignore_nan(correlation, rolling_window)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    ax.plot(time_axis_s, correlation, color="#4C78A8", linewidth=0.35, alpha=0.45)
    ax.plot(time_axis_s, rolling, color="#111111", linewidth=1.6, label=f"Rolling mean ({rolling_window})")
    ax.set_xlabel("Время (s)")
    ax.set_ylabel("Корреляция")
    ax.set_title(f"Корреляция во времени, референсная рефлектограмма {reference_index}")
    ax.grid(alpha=0.25)
    ax.legend()

    saved_paths = save_figure_bundle(fig, path)
    plt.close(fig)
    return saved_paths


def make_abs_delta_lambda_plot(path, time_axis_s, abs_delta_lambda_pm, rolling_window, reference_index):
    rolling = moving_average_ignore_nan(abs_delta_lambda_pm, rolling_window)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    ax.plot(time_axis_s, abs_delta_lambda_pm, color="#F28E2B", linewidth=0.35, alpha=0.45)
    ax.plot(time_axis_s, rolling, color="#111111", linewidth=1.6, label=f"Rolling mean ({rolling_window})")
    ax.set_xlabel("Время (s)")
    ax.set_ylabel(r"|$\Delta \lambda$| (pm)")
    ax.set_title(f"Абсолютный сдвиг длины волны во времени, референсная рефлектограмма {reference_index}")
    ax.grid(alpha=0.25)
    ax.legend()

    saved_paths = save_figure_bundle(fig, path)
    plt.close(fig)
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Рассчитать корреляцию всех рефлектограмм с выбранной референсной рефлектограммой."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Каталог для CSV и PNG результатов",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=64,
        help="Окно для rolling-mean трендов",
    )
    parser.add_argument(
        "--scan-rate",
        type=float,
        default=None,
        help="Необязательная частота записи рефлектограмм в Hz; по умолчанию берётся значение из заголовка файла",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Индекс референсной трассы с нуля",
    )
    parser.add_argument(
        "--corr-to-dlambda-coeff",
        type=float,
        default=2.77e2,
        help="Коэффициент в формуле corr = 1 - coeff * (Delta_lambda_pm)^2",
    )
    parser.add_argument(
        "--ignore-first-meters",
        type=float,
        default=0.0,
        help="Игнорировать столько метров от начала каждой рефлектограммы перед расчётом корреляции",
    )
    parser.add_argument(
        "--parity",
        choices=["all", "even", "odd"],
        default="all",
        help="Использовать все рефлектограммы или только одну чётность",
    )
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    n_reflectograms = int(result["refls_count"])
    if args.reference_index < 0 or args.reference_index >= n_reflectograms:
        raise ValueError(
            f"reference_index must be between 0 and {n_reflectograms - 1}, got {args.reference_index}"
        )

    distance_axis_m, distance_step_m, n_eff = distance_axis_from_sampling_rate(
        result["real_segment_size"],
        result["sampling_rate"],
    )
    keep_mask = distance_axis_m >= float(args.ignore_first_meters)
    if not np.any(keep_mask):
        raise ValueError("ignore_first_meters removes the full reflectogram")
    cropped_data = result["data"][:, keep_mask]
    selected_data, selected_global_indices = select_parity_subset(cropped_data, args.parity)
    if selected_data.shape[0] == 0:
        raise ValueError(f"Parity selection '{args.parity}' left no reflectograms")
    reference_matches = np.flatnonzero(selected_global_indices == int(args.reference_index))
    if reference_matches.size == 0:
        raise ValueError(
            f"reference_index {args.reference_index} is not present in parity subset '{args.parity}'"
        )
    local_reference_index = int(reference_matches[0])

    correlation = correlation_against_reference(
        selected_data,
        reference_index=local_reference_index,
    )
    time_axis_s = selected_global_indices.astype(np.float64) / result["scan_rate"]
    abs_delta_lambda_pm = correlation_to_abs_delta_lambda_pm(
        correlation,
        coeff_pm_inv2=args.corr_to_dlambda_coeff,
    )

    slope, _, total_change = fit_linear_trend(correlation)
    delta_lambda_slope, _, delta_lambda_total_change = fit_linear_trend(abs_delta_lambda_pm)
    correlation_rolling = moving_average_ignore_nan(correlation, args.rolling_window)
    abs_delta_lambda_rolling = moving_average_ignore_nan(abs_delta_lambda_pm, args.rolling_window)

    stem = dat_path.stem
    ref_tag = f"ref_{args.reference_index}"
    scope_tag = ""
    if args.parity != "all":
        scope_tag += f"_{args.parity}"
    if args.ignore_first_meters > 0.0:
        scope_tag += f"_ignore_first_{int(round(args.ignore_first_meters))}m"
    csv_path = output_dir / f"{stem}_corr_vs_{ref_tag}{scope_tag}.csv"
    corr_png_path = output_dir / f"{stem}_corr_vs_{ref_tag}{scope_tag}.png"
    abs_dl_png_path = output_dir / f"{stem}_abs_delta_lambda_pm_vs_{ref_tag}{scope_tag}.png"

    write_csv(csv_path, time_axis_s, correlation, abs_delta_lambda_pm, selected_global_indices)
    corr_saved_paths = make_plot(
        corr_png_path,
        time_axis_s,
        correlation,
        args.rolling_window,
        args.reference_index,
    )
    abs_dl_saved_paths = make_abs_delta_lambda_plot(
        abs_dl_png_path,
        time_axis_s,
        abs_delta_lambda_pm,
        args.rolling_window,
        args.reference_index,
    )
    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=stem,
        ref_tag=f"{ref_tag}{scope_tag}",
        time_axis_s=time_axis_s,
        correlation=correlation,
        correlation_rolling=correlation_rolling,
        abs_delta_lambda_pm=abs_delta_lambda_pm,
        abs_delta_lambda_rolling=abs_delta_lambda_rolling,
        reference_index=args.reference_index,
        scan_rate_hz=result["scan_rate"],
        duration_s=correlation.size / result["scan_rate"],
        coeff_pm_inv2=args.corr_to_dlambda_coeff,
        rolling_window=args.rolling_window,
    )

    print(f"file: {dat_path}")
    print(f"parity: {args.parity}")
    print(f"reference_index: {args.reference_index}")
    print(f"local_reference_index: {local_reference_index}")
    print(f"reflectograms: {correlation.size}")
    print(f"global_index_start: {int(selected_global_indices[0])}")
    print(f"global_index_end: {int(selected_global_indices[-1])}")
    print(f"samples_per_reflectogram: {result['real_segment_size']}")
    print(f"distance_step_m: {distance_step_m:.6f}")
    print(f"total_length_m: {distance_axis_m[-1]:.6f}")
    print(f"ignore_first_meters: {args.ignore_first_meters}")
    print(f"kept_samples: {selected_data.shape[1]}")
    print(f"kept_distance_start_m: {distance_axis_m[keep_mask][0]:.6f}")
    print(f"kept_distance_end_m: {distance_axis_m[keep_mask][-1]:.6f}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"duration_s: {(selected_global_indices[-1] - selected_global_indices[0]) / result['scan_rate']:.6f}")
    print("preprocessing: subtract mean, divide by RMS for each reflectogram")
    print(f"corr_to_dlambda_coeff_pm^-2: {args.corr_to_dlambda_coeff}")
    print(f"mean_correlation: {np.nanmean(correlation):.6f}")
    print(f"std_correlation: {np.nanstd(correlation):.6f}")
    print(f"min_correlation: {np.nanmin(correlation):.6f} at index {int(np.nanargmin(correlation))}")
    print(f"max_correlation: {np.nanmax(correlation):.6f} at index {int(np.nanargmax(correlation))}")
    print(f"linear_slope_per_index: {slope:.12e}")
    print(f"linear_change_over_file: {total_change:.6f}")
    print(f"mean_abs_delta_lambda_pm: {np.nanmean(abs_delta_lambda_pm):.6f}")
    print(f"max_abs_delta_lambda_pm: {np.nanmax(abs_delta_lambda_pm):.6f} at index {int(np.nanargmax(abs_delta_lambda_pm))}")
    print(f"linear_abs_delta_lambda_slope_per_index: {delta_lambda_slope:.12e}")
    print(f"linear_abs_delta_lambda_change_over_file: {delta_lambda_total_change:.6f}")
    print(f"csv_saved_to: {csv_path}")
    print(f"correlation_plot_png_saved_to: {corr_saved_paths['png']}")
    print(f"correlation_plot_svg_saved_to: {corr_saved_paths['svg']}")
    print(f"correlation_plot_pdf_saved_to: {corr_saved_paths['pdf']}")
    print(f"correlation_plot_pickle_saved_to: {corr_saved_paths['pickle']}")
    print(f"abs_delta_lambda_plot_png_saved_to: {abs_dl_saved_paths['png']}")
    print(f"abs_delta_lambda_plot_svg_saved_to: {abs_dl_saved_paths['svg']}")
    print(f"abs_delta_lambda_plot_pdf_saved_to: {abs_dl_saved_paths['pdf']}")
    print(f"abs_delta_lambda_plot_pickle_saved_to: {abs_dl_saved_paths['pickle']}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
