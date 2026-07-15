import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

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


def block_average_rows(data, time_s, block_size):
    data = np.asarray(data, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if data.shape[0] != time_s.size:
        raise ValueError("Row/time size mismatch")
    if int(block_size) <= 1:
        return data.copy(), time_s.copy()
    block_size = int(block_size)
    block_count = data.shape[0] // block_size
    if block_count == 0:
        return data.copy(), time_s.copy()
    kept = block_count * block_size
    data = data[:kept].reshape(block_count, block_size, data.shape[1]).mean(axis=1)
    time_s = time_s[:kept].reshape(block_count, block_size).mean(axis=1)
    return data, time_s


def apply_wavelength_shift_to_field(e_field, delta_lambda_pm, n_eff, lambda0_nm, distance_step_m):
    e_field = np.asarray(e_field, dtype=np.complex128).reshape(-1)
    lambda0_m = float(lambda0_nm) * 1e-9
    delta_lambda_m = float(delta_lambda_pm) * 1e-12
    delta_beta = -2.0 * np.pi * float(n_eff) * delta_lambda_m / (lambda0_m**2)
    idx = np.arange(e_field.size, dtype=np.float64)
    return e_field * np.exp(-1j * 2.0 * delta_beta * idx * float(distance_step_m))


def model_trace_from_field(weights, e_field):
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    e_field = np.asarray(e_field, dtype=np.complex128).reshape(-1)
    pulse_count = weights.size
    coord_count = e_field.size - pulse_count + 1
    if coord_count <= 0:
        raise ValueError("Field is shorter than pulse support")
    trace = np.zeros(coord_count, dtype=np.float64)
    for p in range(1, pulse_count):
        kernel = weights[: pulse_count - p] * weights[p:]
        pair_product = e_field[:-p] * np.conj(e_field[p:])
        trace += 2.0 * np.real(np.correlate(pair_product, kernel, mode="valid"))
    return trace


def candidate_phase_bank(weights, e_base, candidate_index, phase_grid_rad):
    baseline_trace = model_trace_from_field(weights, e_base)
    bank = np.empty((phase_grid_rad.size, baseline_trace.size), dtype=np.float64)
    for row, phase_shift in enumerate(np.asarray(phase_grid_rad, dtype=np.float64)):
        e_mod = e_base.copy()
        e_mod[int(candidate_index)] *= np.exp(1j * float(phase_shift))
        bank[row] = model_trace_from_field(weights, e_mod) - baseline_trace
    return bank


def fit_phase_series(observed_diff, model_bank, phase_grid_rad):
    observed_norm = center_and_rms_normalize_rows(observed_diff)
    model_norm = center_and_rms_normalize_rows(model_bank)
    corr = observed_norm @ model_norm.T / float(observed_norm.shape[1])
    best_idx = np.nanargmax(corr, axis=1)
    best_phase = np.asarray(phase_grid_rad, dtype=np.float64)[best_idx]
    best_corr = corr[np.arange(corr.shape[0]), best_idx]
    return best_phase, best_corr


def unwrap_sorted_phase(time_s, phase_rad):
    order = np.argsort(time_s)
    phase_sorted = np.unwrap(np.asarray(phase_rad, dtype=np.float64)[order])
    out = np.empty_like(phase_sorted)
    out[order] = phase_sorted
    return out


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Candidate score');
plot(data.candidate_distance_m, data.candidate_score, 'o-', 'LineWidth', 1.4);
hold on;
xline(data.best_distance_m, 'r--', 'LineWidth', 1.0);
grid on;
xlabel('Candidate distance (m)');
ylabel('Mean best correlation');
title('Which discrete best explains the post-8.5 s change');

f2 = figure('Color', 'w', 'Name', 'Phase vs time');
plot(data.merged_time_s, data.merged_phase_rad, '.', 'Color', [0.75 0.75 0.75], 'MarkerSize', 7);
hold on;
plot(data.merged_time_s, data.merged_phase_rolling_rad, 'k', 'LineWidth', 1.6);
plot(data.even_time_s, data.even_phase_rad, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 6);
plot(data.odd_time_s, data.odd_phase_rad, '.', 'Color', [1.00 0.50 0.05], 'MarkerSize', 6);
grid on;
xlabel('Time (s)');
ylabel('Phase shift of best discrete (rad)');
title(sprintf('Лучший дискрет в %.3f m', data.best_distance_m));
legend('Merged raw', 'Merged rolling', 'Even', 'Odd', 'Location', 'best');

f3 = figure('Color', 'w', 'Name', 'Fit quality');
plot(data.merged_time_s, data.merged_fit_corr, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 7);
hold on;
plot(data.merged_time_s, data.merged_fit_corr_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Time (s)');
ylabel('Best correlation');
title('Fit quality for the best discrete');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return mat_path, script_path


def main():
    parser = argparse.ArgumentParser(
        description="Найти, какой дискрет около 175-180 m меняет фазу после 8.5 s, и отследить эту фазу во времени."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Каталог для выходных файлов")
    parser.add_argument("--model-mat", default=None, help="MAT-файл из solve_complex_amplitudes_from_harmonics.py")
    parser.add_argument("--scan-rate", type=float, default=None, help="Необязательная частота записи рефлектограмм")
    parser.add_argument("--fiber-z-min", type=float, default=110.0, help="Начало полезного участка волокна в метрах")
    parser.add_argument("--fiber-z-max", type=float, default=350.0, help="Конец полезного участка волокна в метрах")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Вычесть baseline каждой трассы по последним N метрам")
    parser.add_argument("--lambda0-nm", type=float, default=1550.0, help="Центральная длина волны в nm")
    parser.add_argument("--wavelength-shift-pm", type=float, default=0.004, help="Применить этот принятый post-sweep сдвиг длины волны перед анализом пьезо")
    parser.add_argument("--candidate-z-min", type=float, default=175.0, help="Начало поиска движущегося дискрета в метрах")
    parser.add_argument("--candidate-z-max", type=float, default=180.0, help="Конец поиска движущегося дискрета в метрах")
    parser.add_argument("--phase-start-time-s", type=float, default=8.5, help="Примерное время начала фазового воздействия пьезо")
    parser.add_argument("--baseline-duration-s", type=float, default=0.8, help="Длительность спокойного baseline-интервала перед phase-start-time-s")
    parser.add_argument("--phase-grid-size", type=int, default=361, help="Число гипотез фазы от -pi до pi")
    parser.add_argument("--block-size", type=int, default=64, help="Усреднить столько same-parity трасс перед fit-ом")
    parser.add_argument("--rolling-window", type=int, default=9, help="Окно rolling average для итогового фазового тренда")
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.model_mat is None:
        model_mat_path = output_dir / f"{dat_path.stem}_complex_amplitude_factorization_single_sweep_matlab_data.mat"
    else:
        model_mat_path = Path(args.model_mat)
    model = loadmat(model_mat_path)

    e_base = np.asarray(model["E"], dtype=np.complex128).reshape(-1)
    even_weights = np.asarray(model["even_weights"], dtype=np.float64).reshape(-1)
    odd_weights = np.asarray(model["odd_weights"], dtype=np.float64).reshape(-1)
    chain_distance_m = np.asarray(model["chain_distance_m"], dtype=np.float64).reshape(-1)
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
    fiber_data_full = data[:, fiber_mask]
    fiber_distance_m = distance_axis_m[fiber_mask]

    if fiber_data_full.shape[1] != (e_base.size - even_weights.size + 1):
        raise ValueError("Fiber data width does not match model trace width")

    e_shifted = apply_wavelength_shift_to_field(
        e_base,
        delta_lambda_pm=args.wavelength_shift_pm,
        n_eff=n_eff,
        lambda0_nm=args.lambda0_nm,
        distance_step_m=distance_step_m,
    )

    candidate_mask = (chain_distance_m >= float(args.candidate_z_min)) & (chain_distance_m <= float(args.candidate_z_max))
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        raise ValueError("No candidate discretes inside the requested distance window")

    phase_grid_rad = np.linspace(-np.pi, np.pi, int(args.phase_grid_size), dtype=np.float64)
    baseline_start_time_s = float(args.phase_start_time_s) - float(args.baseline_duration_s)

    candidate_scores = []
    candidate_payloads = []
    for candidate_index in candidate_indices:
        parity_payload = {}
        all_fit_corr = []
        for parity, weights in [("even", even_weights), ("odd", odd_weights)]:
            parity_data, parity_global_indices = select_parity_subset(fiber_data_full, parity)
            parity_time_s = parity_global_indices.astype(np.float64) / float(result["scan_rate"])

            baseline_mask = (parity_time_s >= baseline_start_time_s) & (parity_time_s < float(args.phase_start_time_s))
            post_mask = parity_time_s >= float(args.phase_start_time_s)
            if np.count_nonzero(baseline_mask) == 0 or np.count_nonzero(post_mask) == 0:
                raise ValueError(f"Baseline or post window is empty for parity '{parity}'")

            baseline_trace = np.mean(parity_data[baseline_mask], axis=0)
            post_data = parity_data[post_mask]
            post_time_s = parity_time_s[post_mask]
            post_data_avg, post_time_avg_s = block_average_rows(post_data, post_time_s, args.block_size)
            observed_diff = post_data_avg - baseline_trace[None, :]

            model_bank = candidate_phase_bank(weights, e_shifted, candidate_index, phase_grid_rad)
            phase_rad, fit_corr = fit_phase_series(observed_diff, model_bank, phase_grid_rad)
            phase_rad = unwrap_sorted_phase(post_time_avg_s, phase_rad)
            parity_payload[parity] = {
                "time_s": post_time_avg_s,
                "phase_rad": phase_rad,
                "fit_corr": fit_corr,
            }
            all_fit_corr.append(fit_corr)

        score = float(np.mean(np.concatenate(all_fit_corr)))
        candidate_scores.append(score)
        candidate_payloads.append(parity_payload)

    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    best_local_idx = int(np.nanargmax(candidate_scores))
    best_candidate_index = int(candidate_indices[best_local_idx])
    best_distance_m = float(chain_distance_m[best_candidate_index])
    best_payload = candidate_payloads[best_local_idx]

    merged_time_s = np.concatenate([best_payload["even"]["time_s"], best_payload["odd"]["time_s"]])
    merged_phase_rad = np.concatenate([best_payload["even"]["phase_rad"], best_payload["odd"]["phase_rad"]])
    merged_fit_corr = np.concatenate([best_payload["even"]["fit_corr"], best_payload["odd"]["fit_corr"]])
    order = np.argsort(merged_time_s)
    merged_time_s = merged_time_s[order]
    merged_phase_rad = np.unwrap(merged_phase_rad[order])
    merged_fit_corr = merged_fit_corr[order]
    merged_phase_rolling_rad = moving_average_ignore_nan(merged_phase_rad, args.rolling_window)
    merged_fit_corr_rolling = moving_average_ignore_nan(merged_fit_corr, args.rolling_window)

    suffix = "single_discrete_phase_after_8p5s"

    fig1, ax1 = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax1.plot(chain_distance_m[candidate_indices], candidate_scores, "o-", linewidth=1.4)
    ax1.axvline(best_distance_m, color="#D62728", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Координата кандидата (m)")
    ax1.set_ylabel("Средняя лучшая корреляция")
    ax1.set_title("Какой дискрет лучше всего объясняет изменение фазы после 8.5 s")
    ax1.grid(alpha=0.25)
    score_png_path = output_dir / f"{dat_path.stem}_{suffix}_candidate_score.png"
    fig1.savefig(score_png_path, dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax2.plot(merged_time_s, merged_phase_rad, ".", color="#A0A0A0", markersize=3.0, alpha=0.55, label="Объединённые raw")
    ax2.plot(merged_time_s, merged_phase_rolling_rad, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax2.plot(best_payload["even"]["time_s"], best_payload["even"]["phase_rad"], ".", color="#1F77B4", markersize=2.4, alpha=0.55, label="Чётные")
    ax2.plot(best_payload["odd"]["time_s"], best_payload["odd"]["phase_rad"], ".", color="#FF7F0E", markersize=2.4, alpha=0.55, label="Нечётные")
    ax2.set_xlabel("Время (s)")
    ax2.set_ylabel("Фазовый сдвиг лучшего дискрета (rad)")
    ax2.set_title(f"Фаза движущегося дискрета около {best_distance_m:.3f} m")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")
    phase_png_path = output_dir / f"{dat_path.stem}_{suffix}.png"
    fig2.savefig(phase_png_path, dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax3.plot(merged_time_s, merged_fit_corr, ".", color="#4C78A8", markersize=3.0, alpha=0.55, label="Сырые точки")
    ax3.plot(merged_time_s, merged_fit_corr_rolling, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax3.set_xlabel("Время (s)")
    ax3.set_ylabel("Лучшая корреляция")
    ax3.set_title("Качество fit-а для лучшего движущегося дискрета")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")
    fit_png_path = output_dir / f"{dat_path.stem}_{suffix}_fit_quality.png"
    fig3.savefig(fit_png_path, dpi=200)
    plt.close(fig3)

    csv_path = output_dir / f"{dat_path.stem}_{suffix}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("time_s,phase_rad,fit_corr\n")
        for time_s, phase_rad, fit_corr in zip(merged_time_s, merged_phase_rad, merged_fit_corr):
            fout.write(f"{time_s:.10f},{phase_rad:.10f},{fit_corr:.10f}\n")

    mat_path, script_path = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag=suffix,
        payload={
            "candidate_distance_m": chain_distance_m[candidate_indices][:, None],
            "candidate_score": candidate_scores[:, None],
            "best_candidate_index": np.array([[best_candidate_index]], dtype=np.int32),
            "best_distance_m": np.array([[best_distance_m]], dtype=np.float64),
            "merged_time_s": merged_time_s[:, None],
            "merged_phase_rad": merged_phase_rad[:, None],
            "merged_phase_rolling_rad": merged_phase_rolling_rad[:, None],
            "merged_fit_corr": merged_fit_corr[:, None],
            "merged_fit_corr_rolling": merged_fit_corr_rolling[:, None],
            "even_time_s": best_payload["even"]["time_s"][:, None],
            "even_phase_rad": best_payload["even"]["phase_rad"][:, None],
            "odd_time_s": best_payload["odd"]["time_s"][:, None],
            "odd_phase_rad": best_payload["odd"]["phase_rad"][:, None],
            "phase_start_time_s": np.array([[args.phase_start_time_s]], dtype=np.float64),
            "baseline_start_time_s": np.array([[baseline_start_time_s]], dtype=np.float64),
            "wavelength_shift_pm": np.array([[args.wavelength_shift_pm]], dtype=np.float64),
            "baseline_window_start_m": np.array([[baseline_start_m]], dtype=np.float64),
            "baseline_window_end_m": np.array([[baseline_end_m]], dtype=np.float64),
        },
    )

    print(f"file: {dat_path}")
    print(f"model_mat: {model_mat_path}")
    print(f"phase_start_time_s: {args.phase_start_time_s}")
    print(f"baseline_start_time_s: {baseline_start_time_s}")
    print(f"wavelength_shift_pm: {args.wavelength_shift_pm}")
    print(f"candidate_count: {candidate_indices.size}")
    print(f"best_candidate_index: {best_candidate_index}")
    print(f"best_distance_m: {best_distance_m:.6f}")
    print(f"best_candidate_score: {candidate_scores[best_local_idx]:.10f}")
    print(f"merged_phase_min_rad: {np.min(merged_phase_rad):.10f}")
    print(f"merged_phase_max_rad: {np.max(merged_phase_rad):.10f}")
    print(f"merged_fit_corr_mean: {np.mean(merged_fit_corr):.10f}")
    print(f"score_png_saved_to: {score_png_path}")
    print(f"phase_png_saved_to: {phase_png_path}")
    print(f"fit_quality_png_saved_to: {fit_png_path}")
    print(f"csv_saved_to: {csv_path}")
    print(f"matlab_data_saved_to: {mat_path}")
    print(f"matlab_script_saved_to: {script_path}")


if __name__ == "__main__":
    main()
