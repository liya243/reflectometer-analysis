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
    signal,
    signal_rolling,
    coordinate_m,
    coordinate_index,
    scan_rate_hz,
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
            "signal": np.asarray(signal, dtype=np.float64),
            "signal_rolling": np.asarray(signal_rolling, dtype=np.float64),
            "coordinate_m": np.array([[coordinate_m]], dtype=np.float64),
            "coordinate_index": np.array([[coordinate_index]], dtype=np.int32),
            "scan_rate_hz": np.array([[scan_rate_hz]], dtype=np.float64),
            "rolling_window": np.array([[rolling_window]], dtype=np.int32),
        },
    )

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{data_mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Signal at coordinate');
plot(data.time_s, data.signal, 'Color', [0.72 0.80 0.92], 'LineWidth', 0.5);
hold on;
plot(data.time_s, data.signal_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Time (s)');
ylabel('Signal');
title(sprintf('Signal vs time at %.3f m (sample %d)', data.coordinate_m, data.coordinate_index));
legend('Signal', sprintf('Rolling mean (%d)', data.rolling_window), 'Location', 'best');

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
        description="Plot signal vs time at one reflectogram coordinate."
    )
    parser.add_argument("dat_path", help="Path to the .dat file")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory for output files",
    )
    parser.add_argument(
        "--scan-rate",
        type=float,
        default=None,
        help="Optional override for reflectogram scan rate in Hz",
    )
    parser.add_argument(
        "--ignore-first-meters",
        type=float,
        default=0.0,
        help="Ignore this many meters from the beginning when auto-selecting the coordinate",
    )
    parser.add_argument(
        "--coordinate-m",
        type=float,
        default=None,
        help="Coordinate in meters; if omitted, choose the strongest-varying coordinate after ignore-first-meters",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=64,
        help="Rolling mean window in traces",
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

    if args.coordinate_m is None:
        keep_mask = distance_axis_m >= float(args.ignore_first_meters)
        if not np.any(keep_mask):
            raise ValueError("ignore_first_meters removes the full reflectogram")
        std = data[:, keep_mask].std(axis=0)
        local_index = int(np.argmax(std))
        coordinate_index = int(np.where(keep_mask)[0][local_index])
    else:
        coordinate_index = int(np.argmin(np.abs(distance_axis_m - float(args.coordinate_m))))

    coordinate_m = float(distance_axis_m[coordinate_index])
    signal = data[:, coordinate_index]
    time_axis_s = np.arange(signal.size, dtype=np.float64) / result["scan_rate"]
    signal_rolling = moving_average(signal, args.rolling_window)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(time_axis_s, signal, color="#4C78A8", linewidth=0.35, alpha=0.45)
    ax.plot(time_axis_s, signal_rolling, color="#111111", linewidth=1.6, label=f"Rolling mean ({args.rolling_window})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")
    ax.set_title(f"Signal vs time at {coordinate_m:.3f} m (sample {coordinate_index})")
    ax.grid(alpha=0.25)
    ax.legend()

    suffix_tag = f"signal_at_{coordinate_index}"
    png_path = output_dir / f"{dat_path.stem}_{suffix_tag}.png"
    saved_paths = save_figure_bundle(fig, png_path)
    plt.close(fig)

    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag=suffix_tag,
        time_axis_s=time_axis_s,
        signal=signal,
        signal_rolling=signal_rolling,
        coordinate_m=coordinate_m,
        coordinate_index=coordinate_index,
        scan_rate_hz=result["scan_rate"],
        rolling_window=args.rolling_window,
    )

    csv_path = output_dir / f"{dat_path.stem}_{suffix_tag}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("index,time_s,signal\n")
        for idx, (time_s, value) in enumerate(zip(time_axis_s, signal)):
            fout.write(f"{idx},{time_s:.10f},{value:.10f}\n")

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"duration_s: {signal.size / result['scan_rate']:.6f}")
    print(f"distance_step_m: {distance_step_m:.6f}")
    print(f"ignore_first_meters: {args.ignore_first_meters}")
    print(f"coordinate_index: {coordinate_index}")
    print(f"coordinate_m: {coordinate_m:.6f}")
    print(f"signal_mean: {np.mean(signal):.6f}")
    print(f"signal_std: {np.std(signal):.6f}")
    print(f"csv_saved_to: {csv_path}")
    print(f"plot_png_saved_to: {saved_paths['png']}")
    print(f"plot_svg_saved_to: {saved_paths['svg']}")
    print(f"plot_pdf_saved_to: {saved_paths['pdf']}")
    print(f"plot_pickle_saved_to: {saved_paths['pickle']}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
