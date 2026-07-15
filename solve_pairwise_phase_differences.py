import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from raw_data import read_reflectograms, sellmeier_n
from sweep_harmonics_even_odd import (
    build_sweep_intervals,
    detect_reset_times,
    distance_axis_from_sampling_rate,
    harmonics_for_sweeps,
)


def matlab_safe_stem(text):
    safe = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not safe or not safe[0].isalpha():
        safe = f"fig_{safe}"
    return safe


def mean_pulse_weights(data, distance_axis_m, pulse_z_min, pulse_z_max, zero_level_z):
    pulse_mask = (distance_axis_m >= float(pulse_z_min)) & (distance_axis_m <= float(pulse_z_max))
    if not np.any(pulse_mask):
        raise ValueError("Pulse support window is empty")

    zero_index = int(np.argmin(np.abs(distance_axis_m - float(zero_level_z))))
    zeroed = np.asarray(data, dtype=np.float64) - np.asarray(data, dtype=np.float64)[:, [zero_index]]
    pulse_distance_m = distance_axis_m[pulse_mask]
    pulse_zeroed = zeroed[:, pulse_mask]
    even_weights = np.mean(pulse_zeroed[0::2], axis=0)
    odd_weights = np.mean(pulse_zeroed[1::2], axis=0)
    return pulse_distance_m, even_weights, odd_weights, float(distance_axis_m[zero_index]), zero_index


def solve_diagonal_entries(h_even, h_odd, even_weights, odd_weights, ridge_lambda):
    h_even = np.asarray(h_even, dtype=np.complex128)
    h_odd = np.asarray(h_odd, dtype=np.complex128)
    even_weights = np.asarray(even_weights, dtype=np.float64).reshape(-1)
    odd_weights = np.asarray(odd_weights, dtype=np.float64).reshape(-1)

    coord_count, lag_count = h_even.shape
    pulse_count = even_weights.size
    chain_length = coord_count + pulse_count - 1

    solved_diagonals = []
    residual_rms = []
    for lag_idx, p in enumerate(range(1, pulse_count)):
        unknown_count = chain_length - p
        rows = 2 * coord_count
        system = np.zeros((rows, unknown_count), dtype=np.complex128)
        even_kernel = even_weights[: pulse_count - p] * even_weights[p:]
        odd_kernel = odd_weights[: pulse_count - p] * odd_weights[p:]

        for coord_idx in range(coord_count):
            system[coord_idx, coord_idx : coord_idx + even_kernel.size] = even_kernel
            system[coord_count + coord_idx, coord_idx : coord_idx + odd_kernel.size] = odd_kernel

        rhs = np.concatenate([h_even[:, lag_idx], h_odd[:, lag_idx]])
        if ridge_lambda > 0.0:
            ridge = np.sqrt(float(ridge_lambda)) * np.eye(unknown_count, dtype=np.complex128)
            augmented_system = np.vstack([system, ridge])
            augmented_rhs = np.concatenate([rhs, np.zeros(unknown_count, dtype=np.complex128)])
        else:
            augmented_system = system
            augmented_rhs = rhs

        solution, *_ = np.linalg.lstsq(augmented_system, augmented_rhs, rcond=None)
        solved_diagonals.append(solution)
        fit = system @ solution
        residual_rms.append(float(np.sqrt(np.mean(np.abs(fit - rhs) ** 2))))

    max_unknown_count = chain_length - 1
    diagonal_matrix = np.full((max_unknown_count, lag_count), np.nan + 0j, dtype=np.complex128)
    for lag_idx, diag in enumerate(solved_diagonals):
        diagonal_matrix[: diag.size, lag_idx] = diag

    return {
        "chain_length": chain_length,
        "diagonal_matrix": diagonal_matrix,
        "solved_diagonals": solved_diagonals,
        "residual_rms": np.asarray(residual_rms, dtype=np.float64),
    }


def synchronize_phase_chain(solved_diagonals, amplitude_floor):
    lag_count = len(solved_diagonals)
    chain_length = solved_diagonals[0].size + 1 if lag_count > 0 else 0
    for lag_idx, diag in enumerate(solved_diagonals, start=1):
        chain_length = max(chain_length, diag.size + lag_idx)

    matrix = np.zeros((chain_length, chain_length), dtype=np.complex128)
    for idx in range(chain_length):
        matrix[idx, idx] = 1.0

    for p, diag in enumerate(solved_diagonals, start=1):
        amplitude = np.abs(diag)
        valid = amplitude > float(amplitude_floor)
        phase_only = np.zeros_like(diag, dtype=np.complex128)
        phase_only[valid] = diag[valid] / amplitude[valid]
        for i, value in enumerate(phase_only):
            if value == 0.0:
                continue
            weight = amplitude[i]
            matrix[i, i + p] += weight * value
            matrix[i + p, i] += weight * np.conj(value)

    eigvals, eigvecs = np.linalg.eigh(matrix)
    vector = eigvecs[:, np.argmax(eigvals)]
    unit_vector = vector / np.maximum(np.abs(vector), 1e-12)
    phase_chain = np.unwrap(np.angle(unit_vector))
    return unit_vector, phase_chain, matrix


def phase_consistency_rms(unit_vector, solved_diagonals, amplitude_floor):
    errors = []
    for p, diag in enumerate(solved_diagonals, start=1):
        amplitude = np.abs(diag)
        valid = amplitude > float(amplitude_floor)
        measured = np.zeros_like(diag, dtype=np.complex128)
        measured[valid] = diag[valid] / amplitude[valid]
        predicted = unit_vector[:-p] * np.conj(unit_vector[p:])
        local_error = np.abs(predicted[valid] - measured[valid])
        if local_error.size > 0:
            errors.append(np.mean(local_error))
        else:
            errors.append(np.nan)
    return np.asarray(errors, dtype=np.float64)


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Solved pairwise phase differences');
subplot(2,1,1);
imagesc(data.lag_indices, data.chain_distance_m, data.diagonal_phase);
axis xy; colorbar; xlabel('Lag p'); ylabel('Chain distance (m)'); title('arg g_p(i)');
subplot(2,1,2);
imagesc(data.lag_indices, data.chain_distance_m, data.diagonal_amplitude);
axis xy; colorbar; xlabel('Lag p'); ylabel('Chain distance (m)'); title('|g_p(i)|');

f2 = figure('Color', 'w', 'Name', 'Восстановленная фазовая цепочка');
plot(data.chain_distance_m, data.phase_chain, 'LineWidth', 1.6);
grid on; xlabel('Координата цепочки (m)'); ylabel('Восстановленная фаза (rad)');
title('Восстановленная фазовая цепочка с точностью до глобальной фазы');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return {"mat": mat_path, "script": script_path}


def main():
    parser = argparse.ArgumentParser(
        description="Решить линейные системы для g_p(i) = exp(i(phi_i - phi_{i+p})) по чётным/нечётным гармоникам свипа."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Каталог для выходных файлов")
    parser.add_argument("--scan-rate", type=float, default=None, help="Необязательная частота записи рефлектограмм в Hz")
    parser.add_argument("--fiber-z-min", type=float, default=100.0, help="Начало полезного участка волокна в метрах")
    parser.add_argument("--fiber-z-max", type=float, default=280.0, help="Конец полезного участка волокна в метрах")
    parser.add_argument("--pulse-z-min", type=float, default=75.0, help="Начало поддержки импульса в метрах")
    parser.add_argument("--pulse-z-max", type=float, default=85.0, help="Конец поддержки импульса в метрах")
    parser.add_argument("--zero-level-z", type=float, default=70.0, help="Координата нулевого уровня для весов импульса")
    parser.add_argument("--lambda0-nm", type=float, default=1550.0, help="Центральная длина волны в nm")
    parser.add_argument("--sweep-span-pm", type=float, default=0.78, help="Размах одного свипа длины волны в pm")
    parser.add_argument("--rolling-window", type=int, default=64, help="Окно сглаживания детектора сбросов в трассах")
    parser.add_argument("--min-period-s", type=float, default=0.05, help="Минимальный период свипа для детектора сбросов")
    parser.add_argument("--prominence-sigma", type=float, default=3.0, help="Порог детектора сбросов в робастных sigma")
    parser.add_argument("--refine-window-fraction", type=float, default=0.15, help="Окно локального уточнения как доля найденного периода")
    parser.add_argument("--ridge-lambda", type=float, default=1e-6, help="Ridge-регуляризация для линейной МНК-задачи по диагоналям")
    parser.add_argument("--amplitude-floor", type=float, default=1e-4, help="Игнорировать g_p(i) ниже этого порога при синхронизации фаз")
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
    fiber_mask = (distance_axis_m >= float(args.fiber_z_min)) & (distance_axis_m <= float(args.fiber_z_max))
    if not np.any(fiber_mask):
        raise ValueError("Fiber window is empty")

    pulse_distance_m, even_weights, odd_weights, zero_level_actual_m, zero_index = mean_pulse_weights(
        data,
        distance_axis_m,
        pulse_z_min=args.pulse_z_min,
        pulse_z_max=args.pulse_z_max,
        zero_level_z=args.zero_level_z,
    )
    pulse_count = pulse_distance_m.size
    lag_indices = np.arange(1, pulse_count, dtype=np.int64)
    lag_distances_m = lag_indices.astype(np.float64) * float(distance_step_m)
    lambda0_m = float(args.lambda0_nm) * 1e-9
    sweep_span_m = float(args.sweep_span_pm) * 1e-12
    delta_beta_span = -2.0 * np.pi * float(n_eff) * sweep_span_m / (lambda0_m**2)

    mean_complex_harmonics = {}
    fiber_distance_m = distance_axis_m[fiber_mask]
    for parity in ["even", "odd"]:
        fiber_data = data[:, fiber_mask][0::2] if parity == "even" else data[:, fiber_mask][1::2]
        parity_global_indices = np.arange(0, result["refls_count"], 2, dtype=np.int64) if parity == "even" else np.arange(1, result["refls_count"], 2, dtype=np.int64)
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
        sweep_intervals_s = build_sweep_intervals(reset_times_s)
        harmonic_cube, _, _ = harmonics_for_sweeps(
            fiber_data,
            parity_time_s,
            sweep_intervals_s,
            delta_beta_span=delta_beta_span,
            lag_distances_m=lag_distances_m,
        )
        mean_complex_harmonics[parity] = np.mean(harmonic_cube, axis=0)

    solved = solve_diagonal_entries(
        mean_complex_harmonics["even"],
        mean_complex_harmonics["odd"],
        even_weights,
        odd_weights,
        ridge_lambda=args.ridge_lambda,
    )
    unit_vector, phase_chain, sync_matrix = synchronize_phase_chain(
        solved["solved_diagonals"],
        amplitude_floor=args.amplitude_floor,
    )
    consistency_rms = phase_consistency_rms(
        unit_vector,
        solved["solved_diagonals"],
        amplitude_floor=args.amplitude_floor,
    )

    chain_length = solved["chain_length"]
    chain_distance_m = fiber_distance_m[0] + np.arange(chain_length, dtype=np.float64) * float(distance_step_m)
    diagonal_amplitude = np.abs(solved["diagonal_matrix"])
    diagonal_phase = np.angle(solved["diagonal_matrix"])

    diag_fig, diag_axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    im0 = diag_axes[0].imshow(
        diagonal_phase,
        aspect="auto",
        origin="lower",
        cmap="twilight",
        extent=[lag_indices[0], lag_indices[-1], chain_distance_m[0], chain_distance_m[-1]],
    )
    diag_axes[0].set_ylabel("Координата цепочки (m)")
    diag_axes[0].set_title("Решённые попарные фазовые разности: arg g_p(i)")
    diag_fig.colorbar(im0, ax=diag_axes[0], label="Фаза (rad)")
    im1 = diag_axes[1].imshow(
        diagonal_amplitude,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[lag_indices[0], lag_indices[-1], chain_distance_m[0], chain_distance_m[-1]],
    )
    diag_axes[1].set_xlabel("Лаг p")
    diag_axes[1].set_ylabel("Координата цепочки (m)")
    diag_axes[1].set_title("Решённые попарные фазовые разности: |g_p(i)|")
    diag_fig.colorbar(im1, ax=diag_axes[1], label="|g_p(i)|")
    diag_png_path = output_dir / f"{dat_path.stem}_solved_pairwise_phase_differences.png"
    diag_fig.savefig(diag_png_path, dpi=200)
    plt.close(diag_fig)

    phase_fig, phase_axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    phase_axes[0].plot(chain_distance_m, phase_chain, color="#111111", linewidth=1.4)
    phase_axes[0].set_xlabel("Координата цепочки (m)")
    phase_axes[0].set_ylabel("Восстановленная фаза (rad)")
    phase_axes[0].set_title("Восстановленная фазовая цепочка с точностью до глобальной фазы")
    phase_axes[0].grid(alpha=0.25)
    phase_axes[1].plot(lag_indices, solved["residual_rms"], linewidth=1.6, label="RMS невязки линейного решения")
    phase_axes[1].plot(lag_indices, consistency_rms, linewidth=1.6, label="Невязка синхронизации")
    phase_axes[1].set_xlabel("Лаг p")
    phase_axes[1].set_ylabel("Ошибка")
    phase_axes[1].set_title("Согласованность fit-а по лагам")
    phase_axes[1].grid(alpha=0.25)
    phase_axes[1].legend()
    phase_png_path = output_dir / f"{dat_path.stem}_recovered_phase_chain.png"
    phase_fig.savefig(phase_png_path, dpi=200)
    plt.close(phase_fig)

    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag="linear_pairwise_phase_solution",
        payload={
            "fiber_distance_m": fiber_distance_m[:, None],
            "chain_distance_m": chain_distance_m[:, None],
            "pulse_distance_m": pulse_distance_m[:, None],
            "lag_indices": lag_indices[:, None],
            "lag_distances_m": lag_distances_m[:, None],
            "even_weights": even_weights[:, None],
            "odd_weights": odd_weights[:, None],
            "even_mean_complex_harmonics": mean_complex_harmonics["even"],
            "odd_mean_complex_harmonics": mean_complex_harmonics["odd"],
            "diagonal_matrix": solved["diagonal_matrix"],
            "diagonal_amplitude": diagonal_amplitude,
            "diagonal_phase": diagonal_phase,
            "phase_chain": phase_chain[:, None],
            "unit_vector": unit_vector[:, None],
            "residual_rms": solved["residual_rms"][:, None],
            "consistency_rms": consistency_rms[:, None],
            "distance_step_m": np.array([[distance_step_m]], dtype=np.float64),
            "zero_level_actual_m": np.array([[zero_level_actual_m]], dtype=np.float64),
            "zero_level_index": np.array([[zero_index]], dtype=np.int32),
        },
    )

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"distance_step_m: {distance_step_m:.10f}")
    print(f"fiber_coordinate_count: {fiber_distance_m.size}")
    print(f"pulse_discrete_count_N: {pulse_count}")
    print(f"chain_length: {chain_length}")
    print(f"ridge_lambda: {args.ridge_lambda}")
    print(f"amplitude_floor: {args.amplitude_floor}")
    print(f"zero_level_actual_m: {zero_level_actual_m:.6f}")
    print(f"mean_even_weight: {np.mean(even_weights):.10e}")
    print(f"mean_odd_weight: {np.mean(odd_weights):.10e}")
    print(f"residual_rms_mean: {np.nanmean(solved['residual_rms']):.10e}")
    print(f"consistency_rms_mean: {np.nanmean(consistency_rms):.10e}")
    print(f"pairwise_solution_png_saved_to: {diag_png_path}")
    print(f"phase_chain_png_saved_to: {phase_png_path}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
