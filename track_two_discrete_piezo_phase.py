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
        raise ValueError(f"Неподдерживаемая чётность трасс: {parity}")
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
        raise ValueError("Размеры строк данных и массива времени не совпадают")
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
        raise ValueError("Поле короче поддержки импульса")
    trace = np.zeros(coord_count, dtype=np.float64)
    for p in range(1, pulse_count):
        kernel = weights[: pulse_count - p] * weights[p:]
        pair_product = e_field[:-p] * np.conj(e_field[p:])
        trace += 2.0 * np.real(np.correlate(pair_product, kernel, mode="valid"))
    return trace


def pair_phase_delta_bank(weights, e_base, pair_start_index, phase_grid_rad):
    baseline_trace = model_trace_from_field(weights, e_base)
    bank = np.empty((phase_grid_rad.size, baseline_trace.size), dtype=np.float64)
    for row, phase_shift in enumerate(np.asarray(phase_grid_rad, dtype=np.float64)):
        e_mod = e_base.copy()
        e_mod[int(pair_start_index) : int(pair_start_index) + 2] *= np.exp(1j * float(phase_shift))
        bank[row] = model_trace_from_field(weights, e_mod) - baseline_trace
    return bank, baseline_trace


def fit_phase_series(observed_diff, model_bank, phase_grid_rad, trace_mask):
    observed_local = np.asarray(observed_diff, dtype=np.float64)[:, trace_mask]
    model_local = np.asarray(model_bank, dtype=np.float64)[:, trace_mask]
    observed_norm = center_and_rms_normalize_rows(observed_local)
    model_norm = center_and_rms_normalize_rows(model_local)
    corr = observed_norm @ model_norm.T / float(observed_norm.shape[1])
    best_idx = np.nanargmax(corr, axis=1)
    best_phase = np.asarray(phase_grid_rad, dtype=np.float64)[best_idx]
    best_corr = corr[np.arange(corr.shape[0]), best_idx]
    return best_phase, best_corr


def wrapped_phase_delta_rad(a, b):
    return np.angle(np.exp(1j * (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def fit_phase_series_continuous(
    observed_diff,
    model_bank,
    phase_grid_rad,
    trace_mask,
    continuity_lambda,
    zero_prior_lambda,
):
    observed_local = np.asarray(observed_diff, dtype=np.float64)[:, trace_mask]
    model_local = np.asarray(model_bank, dtype=np.float64)[:, trace_mask]
    observed_norm = center_and_rms_normalize_rows(observed_local)
    model_norm = center_and_rms_normalize_rows(model_local)
    corr = observed_norm @ model_norm.T / float(observed_norm.shape[1])
    phase_grid_rad = np.asarray(phase_grid_rad, dtype=np.float64)

    best_idx = np.empty(corr.shape[0], dtype=np.int64)
    first_score = corr[0].copy()
    if float(zero_prior_lambda) > 0.0:
        first_score -= float(zero_prior_lambda) * wrapped_phase_delta_rad(phase_grid_rad, 0.0) ** 2
    best_idx[0] = int(np.nanargmax(first_score))

    for row in range(1, corr.shape[0]):
        transition_penalty = float(continuity_lambda) * wrapped_phase_delta_rad(
            phase_grid_rad,
            phase_grid_rad[best_idx[row - 1]],
        ) ** 2
        best_idx[row] = int(np.nanargmax(corr[row] - transition_penalty))

    best_phase = phase_grid_rad[best_idx]
    best_corr = corr[np.arange(corr.shape[0]), best_idx]
    return best_phase, best_corr


def unwrap_by_time(time_s, phase_rad):
    order = np.argsort(time_s)
    out = np.empty_like(np.asarray(phase_rad, dtype=np.float64))
    out[order] = np.unwrap(np.asarray(phase_rad, dtype=np.float64)[order])
    return out


def align_odd_to_even(even_time_s, even_phase_rad, even_corr, odd_time_s, odd_phase_rad, odd_corr, corr_floor):
    even_time_s = np.asarray(even_time_s, dtype=np.float64)
    odd_time_s = np.asarray(odd_time_s, dtype=np.float64)
    even_phase_rad = np.asarray(even_phase_rad, dtype=np.float64)
    odd_phase_rad = np.asarray(odd_phase_rad, dtype=np.float64)
    even_corr = np.asarray(even_corr, dtype=np.float64)
    odd_corr = np.asarray(odd_corr, dtype=np.float64)

    if even_time_s.size < 2 or odd_time_s.size == 0:
        return odd_phase_rad.copy(), 0.0, 0

    even_order = np.argsort(even_time_s)
    even_time_ordered = even_time_s[even_order]
    even_phase_ordered = even_phase_rad[even_order]
    even_corr_ordered = even_corr[even_order]
    interp_even_phase = np.interp(odd_time_s, even_time_ordered, even_phase_ordered)
    interp_even_corr = np.interp(odd_time_s, even_time_ordered, even_corr_ordered)
    valid = (odd_corr >= float(corr_floor)) & (interp_even_corr >= float(corr_floor))
    if np.count_nonzero(valid) < 8:
        valid = np.isfinite(odd_phase_rad) & np.isfinite(interp_even_phase)
    if np.count_nonzero(valid) == 0:
        return odd_phase_rad.copy(), 0.0, 0

    wrapped_offsets = wrapped_phase_delta_rad(odd_phase_rad[valid], interp_even_phase[valid])
    offset = float(np.angle(np.mean(np.exp(1j * wrapped_offsets))))
    odd_aligned = odd_phase_rad - offset
    branch_shift = 2.0 * np.pi * np.round((interp_even_phase - odd_aligned) / (2.0 * np.pi))
    odd_aligned = odd_aligned + branch_shift
    return odd_aligned, offset, int(np.count_nonzero(valid))


def interp_drift_for_parity(drift_mat, parity, query_time_s, drift_sign):
    time = np.asarray(drift_mat[f"{parity}_time_abs_s"], dtype=np.float64).reshape(-1)
    drift = np.asarray(drift_mat[f"{parity}_delta_lambda_pm"], dtype=np.float64).reshape(-1)
    order = np.argsort(time)
    time = time[order]
    drift = drift[order]
    return float(drift_sign) * np.interp(query_time_s, time, drift, left=drift[0], right=drift[-1])


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)

    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

figure('Color', 'w', 'Name', 'Оценка кандидатных пар');
plot(data.candidate_pair_center_m, data.candidate_score, 'o-', 'LineWidth', 1.4);
hold on;
xline(data.best_pair_center_m, 'r--', 'LineWidth', 1.0);
grid on;
xlabel('Центр пары (m)');
ylabel('Средняя лучшая корреляция');
title('Лучший кандидат: два дискрета под пьезой');

figure('Color', 'w', 'Name', 'Фаза пьезоэлемента во времени');
plot(data.merged_time_s, data.merged_phase_rad, '.', 'Color', [0.72 0.72 0.72], 'MarkerSize', 7);
hold on;
plot(data.merged_time_s, data.merged_phase_rolling_rad, 'k', 'LineWidth', 1.6);
plot(data.even_time_s, data.even_phase_aligned_rad, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 6);
plot(data.odd_time_s, data.odd_phase_aligned_rad, '.', 'Color', [1.00 0.50 0.05], 'MarkerSize', 6);
grid on;
xlabel('Время (s)');
ylabel('Общий фазовый сдвиг двух дискретов (rad)');
title(sprintf('Лучшая пара %.3f - %.3f m', data.best_pair_z1_m, data.best_pair_z2_m));
legend('Объединённые raw', 'Объединённые rolling', 'Чётные', 'Нечётные', 'Location', 'best');

figure('Color', 'w', 'Name', 'Raw-фазы до выравнивания чётности');
plot(data.even_time_s, data.even_phase_raw_rad, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 6);
hold on;
plot(data.odd_time_s, data.odd_phase_raw_rad, '.', 'Color', [1.00 0.50 0.05], 'MarkerSize', 6);
grid on;
xlabel('Время (s)');
ylabel('Raw-фаза (rad)');
title(sprintf('Удалённый offset odd-even: %.3f rad', data.odd_even_phase_offset_rad));
legend('Чётные raw', 'Нечётные raw', 'Location', 'best');

figure('Color', 'w', 'Name', 'Качество fit-а');
plot(data.merged_time_s, data.merged_fit_corr, '.', 'Color', [0.12 0.47 0.71], 'MarkerSize', 7);
hold on;
plot(data.merged_time_s, data.merged_fit_corr_rolling, 'k', 'LineWidth', 1.6);
grid on;
xlabel('Время (s)');
ylabel('Лучшая корреляция');
title('Качество fit-а фазы двух дискретов');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return mat_path, script_path


def main():
    parser = argparse.ArgumentParser(
        description="Найти два соседних дискрета, управляемых пьезоэлементом, и восстановить их общий фазовый сдвиг во времени."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Каталог для выходных файлов")
    parser.add_argument("--model-mat", default=None, help="MAT-файл из solve_complex_amplitudes_from_harmonics.py")
    parser.add_argument("--drift-mat", default=None, help="MAT-файл из wavelength_drift_local_slope_after_sweep.py")
    parser.add_argument("--scan-rate", type=float, default=None, help="Необязательная частота записи рефлектограмм")
    parser.add_argument("--fiber-z-min", type=float, default=110.0, help="Начало полезного участка волокна в метрах")
    parser.add_argument("--fiber-z-max", type=float, default=360.0, help="Конец полезного участка волокна в метрах")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Вычесть базовый уровень по последним N метрам")
    parser.add_argument("--lambda0-nm", type=float, default=1550.0, help="Центральная длина волны в nm")
    parser.add_argument("--drift-sign", type=float, default=1.0, help="Использовать -1, если физически нужно инвертировать знак ранее найденного дрейфа")
    parser.add_argument("--candidate-z-min", type=float, default=230.0, help="Начало окна поиска пары в метрах")
    parser.add_argument("--candidate-z-max", type=float, default=240.0, help="Конец окна поиска пары в метрах")
    parser.add_argument("--phase-start-time-s", type=float, default=5.5, help="Примерное время начала пьезосигнала")
    parser.add_argument("--phase-end-time-s", type=float, default=None, help="Необязательное время окончания отслеживания фазы")
    parser.add_argument("--baseline-duration-s", type=float, default=0.35, help="Длительность спокойного baseline-интервала перед phase-start-time-s")
    parser.add_argument("--phase-grid-size", type=int, default=361, help="Число гипотез фазы на интервале от -pi до pi")
    parser.add_argument("--block-size", type=int, default=32, help="Усреднить столько same-parity трасс перед fit-ом")
    parser.add_argument("--fit-window-half-width-m", type=float, default=18.0, help="Полуширина локального окна fit-а вокруг центра кандидатной пары")
    parser.add_argument("--rolling-window", type=int, default=9, help="Окно rolling average по восстановленной фазе")
    parser.add_argument("--phase-continuity-lambda", type=float, default=0.08, help="Вес штрафа за скачки фазы между соседними временными блоками")
    parser.add_argument("--phase-zero-prior-lambda", type=float, default=0.05, help="Вес штрафа, удерживающего первую post-baseline фазу около нуля")
    parser.add_argument("--parity-align-corr-floor", type=float, default=0.35, help="Использовать точки выше этой корреляции для выравнивания нечётной фазы к чётной")
    args = parser.parse_args()

    dat_path = Path(args.dat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_mat_path = (
        output_dir / f"{dat_path.stem}_complex_amplitude_factorization_single_sweep_matlab_data.mat"
        if args.model_mat is None
        else Path(args.model_mat)
    )
    drift_mat_path = (
        output_dir / f"{dat_path.stem}_wavelength_drift_local_slope_after_last_sweep_matlab_data.mat"
        if args.drift_mat is None
        else Path(args.drift_mat)
    )
    model = loadmat(model_mat_path)
    drift = loadmat(drift_mat_path)

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
        raise ValueError("Окно полезного волокна пустое")
    fiber_data_full = data[:, fiber_mask]
    fiber_distance_m = distance_axis_m[fiber_mask]
    expected_width = e_base.size - even_weights.size + 1
    if fiber_data_full.shape[1] != expected_width:
        raise ValueError(f"Ширина окна волокна {fiber_data_full.shape[1]} не совпадает с шириной модели {expected_width}")

    candidate_mask = (
        (chain_distance_m[:-1] >= float(args.candidate_z_min))
        & (chain_distance_m[1:] <= float(args.candidate_z_max))
    )
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        raise ValueError("В заданном окне нет соседних кандидатных пар")

    phase_grid_rad = np.linspace(-np.pi, np.pi, int(args.phase_grid_size), endpoint=False, dtype=np.float64)
    baseline_start_time_s = float(args.phase_start_time_s) - float(args.baseline_duration_s)

    parity_observed = {}
    for parity in ["even", "odd"]:
        parity_data, parity_global_indices = select_parity_subset(fiber_data_full, parity)
        parity_time_s = parity_global_indices.astype(np.float64) / float(result["scan_rate"])
        baseline_mask = (parity_time_s >= baseline_start_time_s) & (parity_time_s < float(args.phase_start_time_s))
        post_mask = parity_time_s >= float(args.phase_start_time_s)
        if args.phase_end_time_s is not None:
            post_mask &= parity_time_s <= float(args.phase_end_time_s)
        if np.count_nonzero(baseline_mask) == 0 or np.count_nonzero(post_mask) == 0:
            raise ValueError(f"Baseline-окно или post-окно пустое для parity {parity}")

        post_avg, post_time_avg_s = block_average_rows(parity_data[post_mask], parity_time_s[post_mask], args.block_size)
        baseline_avg, baseline_time_avg_s = block_average_rows(
            parity_data[baseline_mask],
            parity_time_s[baseline_mask],
            min(args.block_size, max(1, np.count_nonzero(baseline_mask))),
        )
        parity_observed[parity] = {
            "post_data": post_avg,
            "post_time_s": post_time_avg_s,
            "baseline_data": baseline_avg,
            "baseline_time_s": baseline_time_avg_s,
        }

    parity_context = {}
    for parity, weights in [("even", even_weights), ("odd", odd_weights)]:
        observed = parity_observed[parity]
        baseline_model_rows = []
        for t_s in observed["baseline_time_s"]:
            drift_pm = interp_drift_for_parity(drift, parity, t_s, args.drift_sign)
            e_drift = apply_wavelength_shift_to_field(e_base, drift_pm, n_eff, args.lambda0_nm, distance_step_m)
            baseline_model_rows.append(model_trace_from_field(weights, e_drift))
        baseline_residual = np.mean(observed["baseline_data"] - np.asarray(baseline_model_rows), axis=0)

        post_drift_pm = np.asarray(
            [interp_drift_for_parity(drift, parity, t_s, args.drift_sign) for t_s in observed["post_time_s"]],
            dtype=np.float64,
        )
        rounded_drift = np.round(post_drift_pm, decimals=9)
        unique_drift_pm = np.unique(rounded_drift)
        no_piezo_expected = np.empty_like(observed["post_data"])
        e_by_drift = {}
        trace_by_drift = {}
        for drift_value in unique_drift_pm:
            e_drift = apply_wavelength_shift_to_field(e_base, drift_value, n_eff, args.lambda0_nm, distance_step_m)
            e_by_drift[float(drift_value)] = e_drift
            trace_by_drift[float(drift_value)] = model_trace_from_field(weights, e_drift)
        for row, drift_value in enumerate(rounded_drift):
            no_piezo_expected[row] = trace_by_drift[float(drift_value)] + baseline_residual

        parity_context[parity] = {
            "weights": weights,
            "post_drift_pm": rounded_drift,
            "unique_drift_pm": unique_drift_pm,
            "e_by_drift": e_by_drift,
            "no_piezo_expected": no_piezo_expected,
            "observed_diff": observed["post_data"] - no_piezo_expected,
            "time_s": observed["post_time_s"],
        }

    candidate_scores = []
    candidate_payloads = []
    for pair_start_index in candidate_indices:
        pair_center_m = 0.5 * (chain_distance_m[pair_start_index] + chain_distance_m[pair_start_index + 1])
        trace_mask = np.abs(fiber_distance_m - pair_center_m) <= float(args.fit_window_half_width_m)
        if np.count_nonzero(trace_mask) < 4:
            raise ValueError("Локальное окно трассы для fit-а слишком маленькое")

        payload = {}
        all_corr = []
        for parity, weights in [("even", even_weights), ("odd", odd_weights)]:
            context = parity_context[parity]
            phases = np.empty(context["time_s"].size, dtype=np.float64)
            corrs = np.empty(context["time_s"].size, dtype=np.float64)
            for drift_value in context["unique_drift_pm"]:
                row_mask = context["post_drift_pm"] == drift_value
                e_drift = context["e_by_drift"][float(drift_value)]
                model_bank, no_piezo_model = pair_phase_delta_bank(weights, e_drift, pair_start_index, phase_grid_rad)
                phase_rad, fit_corr = fit_phase_series_continuous(
                    context["observed_diff"][row_mask],
                    model_bank,
                    phase_grid_rad,
                    trace_mask,
                    args.phase_continuity_lambda,
                    args.phase_zero_prior_lambda,
                )
                phases[row_mask] = phase_rad
                corrs[row_mask] = fit_corr

            phases = unwrap_by_time(context["time_s"], phases)
            payload[parity] = {
                "time_s": context["time_s"],
                "phase_rad": phases,
                "fit_corr": corrs,
            }
            all_corr.append(corrs)

        candidate_scores.append(float(np.nanmean(np.concatenate(all_corr))))
        candidate_payloads.append(payload)

    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    best_local = int(np.nanargmax(candidate_scores))
    best_pair_start = int(candidate_indices[best_local])
    best_payload = candidate_payloads[best_local]
    best_z1 = float(chain_distance_m[best_pair_start])
    best_z2 = float(chain_distance_m[best_pair_start + 1])
    best_center = 0.5 * (best_z1 + best_z2)

    odd_phase_aligned, odd_even_phase_offset_rad, parity_alignment_count = align_odd_to_even(
        best_payload["even"]["time_s"],
        best_payload["even"]["phase_rad"],
        best_payload["even"]["fit_corr"],
        best_payload["odd"]["time_s"],
        best_payload["odd"]["phase_rad"],
        best_payload["odd"]["fit_corr"],
        args.parity_align_corr_floor,
    )
    best_payload["odd"]["phase_aligned_rad"] = odd_phase_aligned
    best_payload["even"]["phase_aligned_rad"] = best_payload["even"]["phase_rad"].copy()

    merged_time_s = np.concatenate([best_payload["even"]["time_s"], best_payload["odd"]["time_s"]])
    merged_phase_rad = np.concatenate([best_payload["even"]["phase_aligned_rad"], best_payload["odd"]["phase_aligned_rad"]])
    merged_fit_corr = np.concatenate([best_payload["even"]["fit_corr"], best_payload["odd"]["fit_corr"]])
    order = np.argsort(merged_time_s)
    merged_time_s = merged_time_s[order]
    merged_phase_rad = np.unwrap(merged_phase_rad[order])
    merged_fit_corr = merged_fit_corr[order]
    merged_phase_rolling_rad = moving_average_ignore_nan(merged_phase_rad, args.rolling_window)
    merged_fit_corr_rolling = moving_average_ignore_nan(merged_fit_corr, args.rolling_window)

    suffix = "two_discrete_piezo_phase_drift_corrected"
    candidate_center_m = 0.5 * (chain_distance_m[candidate_indices] + chain_distance_m[candidate_indices + 1])

    fig1, ax1 = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax1.plot(candidate_center_m, candidate_scores, "o-", linewidth=1.4)
    ax1.axvline(best_center, color="#D62728", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Центр кандидатной пары (m)")
    ax1.set_ylabel("Средняя лучшая корреляция")
    ax1.set_title("Какие два соседних дискрета лучше всего объясняют пьезосигнал")
    ax1.grid(alpha=0.25)
    score_png_path = output_dir / f"{dat_path.stem}_{suffix}_candidate_score.png"
    fig1.savefig(score_png_path, dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax2.plot(merged_time_s, merged_phase_rad, ".", color="#A8A8A8", markersize=4.0, alpha=0.55, label="Объединённые raw")
    ax2.plot(merged_time_s, merged_phase_rolling_rad, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax2.plot(best_payload["even"]["time_s"], best_payload["even"]["phase_aligned_rad"], ".", color="#1F77B4", markersize=3.2, alpha=0.70, label="Чётные")
    ax2.plot(best_payload["odd"]["time_s"], best_payload["odd"]["phase_aligned_rad"], ".", color="#FF7F0E", markersize=3.2, alpha=0.70, label="Нечётные")
    ax2.set_xlabel("Время (s)")
    ax2.set_ylabel("Общий фазовый сдвиг двух дискретов (rad)")
    ax2.set_title(f"Фаза пьезоэлемента для пары {best_z1:.3f} m и {best_z2:.3f} m")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")
    phase_png_path = output_dir / f"{dat_path.stem}_{suffix}.png"
    fig2.savefig(phase_png_path, dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax3.plot(merged_time_s, merged_fit_corr, ".", color="#4C78A8", markersize=4.0, alpha=0.55, label="Сырые точки")
    ax3.plot(merged_time_s, merged_fit_corr_rolling, color="#111111", linewidth=1.6, label=f"Rolling ({args.rolling_window})")
    ax3.set_xlabel("Время (s)")
    ax3.set_ylabel("Лучшая корреляция")
    ax3.set_title("Качество fit-а для лучшей двухдискретной пьезо-пары")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")
    quality_png_path = output_dir / f"{dat_path.stem}_{suffix}_fit_quality.png"
    fig3.savefig(quality_png_path, dpi=200)
    plt.close(fig3)

    csv_path = output_dir / f"{dat_path.stem}_{suffix}.csv"
    with csv_path.open("w", encoding="utf-8") as fout:
        fout.write("time_s,phase_rad,fit_corr\n")
        for row in zip(merged_time_s, merged_phase_rad, merged_fit_corr):
            fout.write(f"{row[0]:.10f},{row[1]:.10f},{row[2]:.10f}\n")

    mat_path, script_path = save_matlab_bundle(
        output_dir,
        dat_path.stem,
        suffix,
        {
            "candidate_pair_start_index": candidate_indices[:, None].astype(np.int32),
            "candidate_pair_z1_m": chain_distance_m[candidate_indices][:, None],
            "candidate_pair_z2_m": chain_distance_m[candidate_indices + 1][:, None],
            "candidate_pair_center_m": candidate_center_m[:, None],
            "candidate_score": candidate_scores[:, None],
            "best_pair_start_index": np.array([[best_pair_start]], dtype=np.int32),
            "best_pair_z1_m": np.array([[best_z1]], dtype=np.float64),
            "best_pair_z2_m": np.array([[best_z2]], dtype=np.float64),
            "best_pair_center_m": np.array([[best_center]], dtype=np.float64),
            "merged_time_s": merged_time_s[:, None],
            "merged_phase_rad": merged_phase_rad[:, None],
            "merged_phase_rolling_rad": merged_phase_rolling_rad[:, None],
            "merged_fit_corr": merged_fit_corr[:, None],
            "merged_fit_corr_rolling": merged_fit_corr_rolling[:, None],
            "even_time_s": best_payload["even"]["time_s"][:, None],
            "even_phase_rad": best_payload["even"]["phase_aligned_rad"][:, None],
            "even_phase_aligned_rad": best_payload["even"]["phase_aligned_rad"][:, None],
            "even_phase_raw_rad": best_payload["even"]["phase_rad"][:, None],
            "even_fit_corr": best_payload["even"]["fit_corr"][:, None],
            "odd_time_s": best_payload["odd"]["time_s"][:, None],
            "odd_phase_rad": best_payload["odd"]["phase_aligned_rad"][:, None],
            "odd_phase_aligned_rad": best_payload["odd"]["phase_aligned_rad"][:, None],
            "odd_phase_raw_rad": best_payload["odd"]["phase_rad"][:, None],
            "odd_fit_corr": best_payload["odd"]["fit_corr"][:, None],
            "odd_even_phase_offset_rad": np.array([[odd_even_phase_offset_rad]], dtype=np.float64),
            "parity_alignment_count": np.array([[parity_alignment_count]], dtype=np.int32),
            "phase_start_time_s": np.array([[args.phase_start_time_s]], dtype=np.float64),
            "baseline_start_time_s": np.array([[baseline_start_time_s]], dtype=np.float64),
            "baseline_window_start_m": np.array([[baseline_start_m]], dtype=np.float64),
            "baseline_window_end_m": np.array([[baseline_end_m]], dtype=np.float64),
            "fit_window_half_width_m": np.array([[args.fit_window_half_width_m]], dtype=np.float64),
            "drift_sign": np.array([[args.drift_sign]], dtype=np.float64),
            "phase_continuity_lambda": np.array([[args.phase_continuity_lambda]], dtype=np.float64),
            "phase_zero_prior_lambda": np.array([[args.phase_zero_prior_lambda]], dtype=np.float64),
        },
    )

    print(f"file: {dat_path}")
    print(f"model_mat: {model_mat_path}")
    print(f"drift_mat: {drift_mat_path}")
    print(f"phase_start_time_s: {args.phase_start_time_s}")
    print(f"baseline_start_time_s: {baseline_start_time_s}")
    print(f"candidate_count: {candidate_indices.size}")
    print(f"best_pair_start_index: {best_pair_start}")
    print(f"best_pair_z1_m: {best_z1:.6f}")
    print(f"best_pair_z2_m: {best_z2:.6f}")
    print(f"best_pair_center_m: {best_center:.6f}")
    print(f"best_candidate_score: {candidate_scores[best_local]:.10f}")
    print(f"odd_even_phase_offset_rad: {odd_even_phase_offset_rad:.10f}")
    print(f"parity_alignment_count: {parity_alignment_count}")
    print(f"phase_rad_min: {np.nanmin(merged_phase_rad):.10f}")
    print(f"phase_rad_max: {np.nanmax(merged_phase_rad):.10f}")
    print(f"phase_rad_ptp: {np.nanmax(merged_phase_rad) - np.nanmin(merged_phase_rad):.10f}")
    print(f"fit_corr_mean: {np.nanmean(merged_fit_corr):.10f}")
    print(f"fit_corr_min: {np.nanmin(merged_fit_corr):.10f}")
    print(f"score_png_saved_to: {score_png_path}")
    print(f"phase_png_saved_to: {phase_png_path}")
    print(f"fit_quality_png_saved_to: {quality_png_path}")
    print(f"csv_saved_to: {csv_path}")
    print(f"matlab_data_saved_to: {mat_path}")
    print(f"matlab_script_saved_to: {script_path}")


if __name__ == "__main__":
    main()
