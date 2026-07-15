import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from analysis_output_utils import cleanup_outputs_for_dataset, matlab_safe_stem
from raw_data import read_reflectograms
from reflectometer_utils import distance_axis_from_sampling_rate, subtract_trace_baseline_from_tail
from solve_complex_amplitudes_from_harmonics import (
    alternating_rank1_hermitian,
    build_observation_lists,
    canonicalize_global_phase,
    factorization_error_by_lag,
    fit_field_directly_to_harmonics,
    solve_log_magnitudes,
    solve_recursive_phases,
)
from solve_pairwise_phase_differences import mean_pulse_weights, solve_diagonal_entries
from sweep_harmonics_even_odd import build_sweep_intervals, detect_reset_times, harmonics_for_sweeps


def align_global_phase_to_reference(reference_e, candidate_e):
    reference_e = np.asarray(reference_e, dtype=np.complex128).reshape(-1)
    candidate_e = np.asarray(candidate_e, dtype=np.complex128).reshape(-1)
    common = min(reference_e.size, candidate_e.size)
    if common == 0:
        return canonicalize_global_phase(candidate_e)
    ref = reference_e[:common]
    cand = candidate_e[:common]
    weights = np.abs(ref) * np.abs(cand)
    overlap = np.sum(weights * ref * np.conj(cand))
    if abs(overlap) > 1e-12:
        candidate_e = candidate_e * np.exp(1j * np.angle(overlap))
    return canonicalize_global_phase(candidate_e)


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)
    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Complex amplitudes over sweeps');
subplot(2,1,1);
imagesc(data.chain_distance_m, data.sweep_index, data.E_amplitude_over_sweeps);
axis xy; colorbar; xlabel('Distance (m)'); ylabel('Sweep index'); title('|E(z)|');
subplot(2,1,2);
imagesc(data.chain_distance_m, data.sweep_index, data.E_phase_over_sweeps);
axis xy; colorbar; xlabel('Distance (m)'); ylabel('Sweep index'); title('arg(E(z))');

f2 = figure('Color', 'w', 'Name', 'Fit quality over sweeps');
plot(data.sweep_index, data.direct_fit_residual_rms, 'LineWidth', 1.4);
hold on;
plot(data.sweep_index, data.factorization_error_mean, 'LineWidth', 1.4);
grid on; xlabel('Sweep index'); ylabel('Error');
legend('Direct fit residual RMS', 'Mean factorization error', 'Location', 'best');
title('Fit quality by sweep');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return {"mat": mat_path, "script": script_path}


def main():
    parser = argparse.ArgumentParser(
        description="Восстановить векторы комплексных амплитуд E(z) для каждого свипа и отследить их эволюцию."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Каталог для выходных файлов")
    parser.add_argument("--cleanup-dataset-outputs", action="store_true", help="Удалить предыдущие результаты этого датасета перед повторным запуском")
    parser.add_argument("--scan-rate", type=float, default=None, help="Необязательная частота записи рефлектограмм в Hz")
    parser.add_argument("--fiber-z-min", type=float, default=105.0, help="Начало полезного участка волокна в метрах")
    parser.add_argument("--fiber-z-max", type=float, default=280.0, help="Конец полезного участка волокна в метрах")
    parser.add_argument("--pulse-z-min", type=float, default=75.0, help="Начало поддержки импульса в метрах")
    parser.add_argument("--pulse-z-max", type=float, default=85.0, help="Конец поддержки импульса в метрах")
    parser.add_argument("--zero-level-z", type=float, default=70.0, help="Координата нулевого уровня для весов импульса")
    parser.add_argument("--lambda0-nm", type=float, default=1550.0, help="Центральная длина волны в nm")
    parser.add_argument("--sweep-span-pm", type=float, default=3.125, help="Размах одного свипа длины волны в pm")
    parser.add_argument("--rolling-window", type=int, default=64, help="Окно сглаживания детектора сбросов в трассах")
    parser.add_argument("--min-period-s", type=float, default=0.05, help="Минимальный период свипа для детектора сбросов")
    parser.add_argument("--prominence-sigma", type=float, default=2.0, help="Порог детектора сбросов в робастных sigma")
    parser.add_argument("--refine-window-fraction", type=float, default=0.15, help="Окно локального уточнения как доля найденного периода")
    parser.add_argument("--reset-time-shift-ms", type=float, default=3.0, help="Сдвинуть найденные времена сбросов позже на это число ms")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Вычесть baseline каждой трассы по последним N метрам")
    parser.add_argument("--ridge-lambda", type=float, default=1e-6, help="Ridge-регуляризация для линейной МНК-задачи по диагоналям")
    parser.add_argument("--amplitude-floor", type=float, default=1e-4, help="Игнорировать попарные произведения ниже этого порога")
    parser.add_argument("--als-iters", type=int, default=20, help="Число ALS-итераций для rank-1 инициализации")
    parser.add_argument("--lag-min", type=int, default=2, help="Минимальный лаг p для прямого fit-а")
    parser.add_argument("--lag-max", type=int, default=16, help="Максимальный лаг p для прямого fit-а")
    parser.add_argument("--direct-iters", type=int, default=20, help="Число ALS-итераций прямого fit-а к измеренным гармоникам")
    parser.add_argument("--direct-damping", type=float, default=0.25, help="Коэффициент damping-а для прямых ALS-обновлений")
    parser.add_argument("--direct-ridge-lambda", type=float, default=1e-3, help="Ridge-регуляризация для каждого скалярного ALS-обновления")
    parser.add_argument("--max-sweeps", type=int, default=None, help="Необязательное ограничение числа обрабатываемых свипов")
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.cleanup_dataset_outputs:
        removed = cleanup_outputs_for_dataset(output_dir, dat_path.stem)
        print(f"cleanup_removed_count: {len(removed)}")

    result = read_reflectograms(str(dat_path), scan_rate=args.scan_rate)
    data = np.asarray(result["data"], dtype=np.float64)
    distance_axis_m, distance_step_m, n_eff = distance_axis_from_sampling_rate(
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

    pulse_distance_m, even_weights, odd_weights, zero_level_actual_m, zero_index = mean_pulse_weights(
        data,
        distance_axis_m,
        pulse_z_min=args.pulse_z_min,
        pulse_z_max=args.pulse_z_max,
        zero_level_z=args.zero_level_z,
    )
    pulse_count = pulse_distance_m.size
    lag_indices = np.arange(1, pulse_count, dtype=np.int64)
    lag_max = min(int(args.lag_max), int(lag_indices[-1]))
    lag_mask = (lag_indices >= int(args.lag_min)) & (lag_indices <= lag_max)
    if not np.any(lag_mask):
        raise ValueError("Selected lag window is empty")
    fit_lag_indices = lag_indices[lag_mask]
    lag_distances_m = lag_indices.astype(np.float64) * float(distance_step_m)
    lambda0_m = float(args.lambda0_nm) * 1e-9
    sweep_span_m = float(args.sweep_span_pm) * 1e-12
    delta_beta_span = -2.0 * np.pi * float(n_eff) * sweep_span_m / (lambda0_m**2)

    harmonic_cubes = {}
    dominant_periods = {}
    reset_counts = {}
    for parity in ["even", "odd"]:
        fiber_data = data[:, fiber_mask][0::2] if parity == "even" else data[:, fiber_mask][1::2]
        parity_global_indices = (
            np.arange(0, result["refls_count"], 2, dtype=np.int64)
            if parity == "even"
            else np.arange(1, result["refls_count"], 2, dtype=np.int64)
        )
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
        reset_times_s = reset_times_s + 1e-3 * float(args.reset_time_shift_ms)
        sweep_intervals_s = build_sweep_intervals(reset_times_s)
        harmonic_cube, _, _ = harmonics_for_sweeps(
            fiber_data,
            parity_time_s,
            sweep_intervals_s,
            delta_beta_span=delta_beta_span,
            lag_distances_m=lag_distances_m,
        )
        harmonic_cubes[parity] = harmonic_cube
        dominant_periods[parity] = dominant_period_s
        reset_counts[parity] = reset_times_s.size

    sweep_count = min(harmonic_cubes["even"].shape[0], harmonic_cubes["odd"].shape[0])
    if args.max_sweeps is not None:
        sweep_count = min(sweep_count, int(args.max_sweeps))
    if sweep_count <= 0:
        raise ValueError("No common complete sweeps remain")

    previous_e = None
    amplitude_rows = []
    phase_rows = []
    residual_rms = []
    factorization_error_mean = []

    for sweep_index in range(sweep_count):
        selected_even = harmonic_cubes["even"][sweep_index]
        selected_odd = harmonic_cubes["odd"][sweep_index]

        solved = solve_diagonal_entries(
            selected_even,
            selected_odd,
            even_weights,
            odd_weights,
            ridge_lambda=args.ridge_lambda,
        )
        obs = build_observation_lists(solved["solved_diagonals"], args.amplitude_floor)

        if previous_e is None or previous_e.size != obs["chain_length"]:
            magnitudes0 = solve_log_magnitudes(obs, obs["chain_length"], ridge_lambda=args.ridge_lambda)
            phases0 = solve_recursive_phases(obs, obs["chain_length"])
            x0 = magnitudes0 * np.exp(1j * phases0)
            initial_e = alternating_rank1_hermitian(obs, obs["chain_length"], x0=x0, n_iters=args.als_iters)
        else:
            initial_e = previous_e.copy()

        direct_fit = fit_field_directly_to_harmonics(
            even_weights=even_weights,
            odd_weights=odd_weights,
            even_harmonics=selected_even[:, lag_mask],
            odd_harmonics=selected_odd[:, lag_mask],
            lag_indices=fit_lag_indices,
            initial_e=initial_e,
            n_iters=args.direct_iters,
            damping=args.direct_damping,
            ridge_lambda=args.direct_ridge_lambda,
        )
        recovered_e = direct_fit["field"]
        if previous_e is not None:
            recovered_e = align_global_phase_to_reference(previous_e, recovered_e)
        previous_e = recovered_e.copy()

        chain_distance_m = fiber_distance_m[0] + np.arange(recovered_e.size, dtype=np.float64) * float(distance_step_m)
        amplitude_rows.append(np.abs(recovered_e))
        phase_rows.append(np.unwrap(np.angle(recovered_e)))
        residual_rms.append(float(np.sqrt(np.mean(direct_fit["residual_vector"] ** 2))))
        factorization_error_mean.append(
            float(np.nanmean(factorization_error_by_lag(recovered_e, solved["solved_diagonals"], args.amplitude_floor)))
        )

        print(
            f"sweep {sweep_index + 1}/{sweep_count}: "
            f"direct_fit_residual_rms={residual_rms[-1]:.6e}, "
            f"factorization_error_mean={factorization_error_mean[-1]:.6e}"
        )

    amplitude_matrix = np.stack(amplitude_rows, axis=0)
    phase_matrix = np.stack(phase_rows, axis=0)
    sweep_axis = np.arange(sweep_count, dtype=np.int64)

    amp_fig, amp_ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    im_amp = amp_ax.imshow(
        amplitude_matrix,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[chain_distance_m[0], chain_distance_m[-1], sweep_axis[0], sweep_axis[-1]],
    )
    amp_ax.set_xlabel("Расстояние (m)")
    amp_ax.set_ylabel("Индекс свипа")
    amp_ax.set_title("Восстановленное |E(z)| по свипам")
    amp_fig.colorbar(im_amp, ax=amp_ax, label="|E|")
    amp_png_path = output_dir / f"{dat_path.stem}_complex_amplitude_magnitude_over_sweeps.png"
    amp_fig.savefig(amp_png_path, dpi=200)
    plt.close(amp_fig)

    phase_fig, phase_ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    im_phase = phase_ax.imshow(
        phase_matrix,
        aspect="auto",
        origin="lower",
        cmap="twilight",
        extent=[chain_distance_m[0], chain_distance_m[-1], sweep_axis[0], sweep_axis[-1]],
    )
    phase_ax.set_xlabel("Расстояние (m)")
    phase_ax.set_ylabel("Индекс свипа")
    phase_ax.set_title("Восстановленное arg(E(z)) по свипам")
    phase_fig.colorbar(im_phase, ax=phase_ax, label="Фаза (rad)")
    phase_png_path = output_dir / f"{dat_path.stem}_complex_amplitude_phase_over_sweeps.png"
    phase_fig.savefig(phase_png_path, dpi=200)
    plt.close(phase_fig)

    quality_fig, quality_ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    quality_ax.plot(sweep_axis, residual_rms, linewidth=1.5, label="RMS невязки прямого fit-а")
    quality_ax.plot(sweep_axis, factorization_error_mean, linewidth=1.5, label="Средняя ошибка факторизации")
    quality_ax.set_xlabel("Индекс свипа")
    quality_ax.set_ylabel("Ошибка")
    quality_ax.set_title("Качество fit-а по свипам")
    quality_ax.grid(alpha=0.25)
    quality_ax.legend()
    quality_png_path = output_dir / f"{dat_path.stem}_complex_amplitude_fit_quality_over_sweeps.png"
    quality_fig.savefig(quality_png_path, dpi=200)
    plt.close(quality_fig)

    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag="complex_amplitudes_over_sweeps",
        payload={
            "chain_distance_m": chain_distance_m[:, None],
            "pulse_distance_m": pulse_distance_m[:, None],
            "lag_indices": lag_indices[:, None],
            "fit_lag_indices": fit_lag_indices[:, None],
            "sweep_index": sweep_axis[:, None],
            "E_amplitude_over_sweeps": amplitude_matrix,
            "E_phase_over_sweeps": phase_matrix,
            "direct_fit_residual_rms": np.asarray(residual_rms, dtype=np.float64)[:, None],
            "factorization_error_mean": np.asarray(factorization_error_mean, dtype=np.float64)[:, None],
            "distance_step_m": np.array([[distance_step_m]], dtype=np.float64),
            "baseline_window_start_m": np.array([[baseline_start_m]], dtype=np.float64),
            "baseline_window_end_m": np.array([[baseline_end_m]], dtype=np.float64),
            "zero_level_actual_m": np.array([[zero_level_actual_m]], dtype=np.float64),
            "zero_level_index": np.array([[zero_index]], dtype=np.int32),
        },
    )

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"fiber_distance_start_m: {fiber_distance_m[0]:.6f}")
    print(f"fiber_distance_end_m: {fiber_distance_m[-1]:.6f}")
    print(f"pulse_discrete_count_N: {pulse_count}")
    print(f"sweep_count_processed: {sweep_count}")
    print(f"lag_min: {args.lag_min}")
    print(f"lag_max: {lag_max}")
    print(f"direct_damping: {args.direct_damping}")
    print(f"direct_ridge_lambda: {args.direct_ridge_lambda}")
    print(f"mean_direct_fit_residual_rms: {np.mean(residual_rms):.10e}")
    print(f"mean_factorization_error: {np.mean(factorization_error_mean):.10e}")
    print(f"amplitude_heatmap_png_saved_to: {amp_png_path}")
    print(f"phase_heatmap_png_saved_to: {phase_png_path}")
    print(f"fit_quality_png_saved_to: {quality_png_path}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
