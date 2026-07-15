"""Восстановление комплексных амплитуд дискретных отражателей по гармоникам свипа.

Это центральный скрипт деконволюции в проекте. Он использует один калиброванный
свип длины волны, чтобы восстановить комплексное дискретное поле

    E_i = A_i * exp(i phi_i)

вдоль волокна. Алгоритм:

1. Разделить чётные и нечётные рефлектограммы, потому что длины импульса различаются.
2. Найти или принять заданные границы сброса пилообразного свипа.
3. Аппроксимировать гармоники Фурье H_p(z) на выбранном свипе длины волны.
4. Использовать известные формы чётного и нечётного импульсов, чтобы деконволюцией
   перевести H_p(z) в попарные произведения
   X_i,p ~= E_i * conj(E_{i+p}).
5. Факторизовать эти произведения как эрмитову матрицу ранга 1: E E^H.
6. Уточнить E прямым fit-ом к измеренным картам H_p(z).

Восстановленное поле определено только с точностью до глобальной фазы. Скрипт
фиксирует эту калибровку, поворачивая первый комплексный отсчёт так, чтобы он
был вещественным и положительным. Полный вывод и ограничения описаны в
docs/deconvolution_algorithm.md.
"""

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
from solve_pairwise_phase_differences import mean_pulse_weights, solve_diagonal_entries
from sweep_harmonics_even_odd import build_sweep_intervals, detect_reset_times, harmonics_for_sweeps


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
        raise ValueError("Переопределённый период сбросов должен быть положительным")

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


def build_observation_lists(solved_diagonals, amplitude_floor):
    """Преобразовать диагонали попарных произведений в взвешенные наблюдения графа.

    После линейной деконволюции гармоник есть шумные оценки

        X_i,p = E_i * conj(E_{i+p})

    для нескольких лагов p. Удобно считать каждый валидный элемент ребром графа:
    узел i соединён с узлом j=i+p, а измеренное значение ребра равно комплексному
    произведению. Дальше восстановление ищет такие значения узлов E_i, которые
    лучше всего объясняют все эти рёбра.

    Очень малые |X_i,p| имеют почти случайную фазу, поэтому `amplitude_floor`
    отбрасывает их до того, как они дестабилизируют восстановление фаз.
    """
    obs_i = []
    obs_j = []
    obs_value = []
    obs_weight = []
    chain_length = 0
    for p, diag in enumerate(solved_diagonals, start=1):
        chain_length = max(chain_length, diag.size + p)
        amplitude = np.abs(diag)
        valid = np.isfinite(diag) & (amplitude > float(amplitude_floor))
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        obs_i.append(idx)
        obs_j.append(idx + p)
        obs_value.append(diag[idx])
        obs_weight.append(amplitude[idx])
    if len(obs_i) == 0:
        raise ValueError("После amplitude_floor не осталось ни одного элемента диагоналей")
    return {
        "i": np.concatenate(obs_i).astype(np.int64),
        "j": np.concatenate(obs_j).astype(np.int64),
        "value": np.concatenate(obs_value).astype(np.complex128),
        "weight": np.concatenate(obs_weight).astype(np.float64),
        "chain_length": int(chain_length),
    }


def solve_log_magnitudes(obs, chain_length, ridge_lambda, eps=1e-12):
    """Оценить |E_i| по модулям попарных произведений.

    Так как

        |E_i * conj(E_j)| = |E_i| * |E_j|,

    логарифм превращает задачу для амплитуд в линейную систему:

        log |X_ij| = log |E_i| + log |E_j|.

    Последняя дополнительная строка — слабая привязка, которая не даёт почти
    сингулярному графу уплывать по общему масштабу в плохо связанных областях.
    """
    rows = obs["value"].size
    system = np.zeros((rows + 1, chain_length), dtype=np.float64)
    rhs = np.zeros(rows + 1, dtype=np.float64)
    weights = np.sqrt(np.maximum(obs["weight"], eps))
    for row, (i_idx, j_idx, value, row_weight) in enumerate(zip(obs["i"], obs["j"], obs["value"], weights)):
        system[row, i_idx] = row_weight
        system[row, j_idx] = row_weight
        rhs[row] = row_weight * np.log(max(abs(value), eps))
    # Слабая привязка масштаба на случай, если граф наблюдений почти сингулярен.
    system[-1, 0] = np.sqrt(float(ridge_lambda) + eps)
    rhs[-1] = 0.0
    solution, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    return np.exp(solution)


def solve_recursive_phases(obs, chain_length):
    """Построить грубое начальное приближение фаз по соседним произведениям.

    Если X_i,1 = E_i * conj(E_{i+1}), то

        arg(X_i,1) = phase(E_i) - phase(E_{i+1}).

    Это даёт phase(E_{i+1}) рекурсивно. Если соседнее произведение отсутствует,
    фаза просто продолжается предыдущим значением; последующие ALS-этапы уточняют
    это грубое приближение по всем доступным лагам.
    """
    neighbor_products = {}
    for i_idx, j_idx, value in zip(obs["i"], obs["j"], obs["value"]):
        if j_idx == i_idx + 1:
            neighbor_products[i_idx] = value
    phases = np.zeros(chain_length, dtype=np.float64)
    for idx in range(chain_length - 1):
        if idx in neighbor_products:
            phases[idx + 1] = phases[idx] - np.angle(neighbor_products[idx])
        else:
            phases[idx + 1] = phases[idx]
    return phases


def alternating_rank1_hermitian(obs, chain_length, x0, n_iters):
    """Факторизовать шумные попарные произведения в оценку поля ранга 1.

    Идеальная матрица всех попарных произведений:

        X_ij = E_i * conj(E_j) = E E^H.

    Измерены только некоторые диагонали этой матрицы, причём с шумом. Этот цикл
    попеременного МНК обновляет один комплексный отсчёт E_n за раз, удерживая
    остальные отсчёты фиксированными. При фиксированных соседях взвешенное
    МНК-обновление имеет простой замкнутый вид.

    Результат используется как стабильное начальное приближение для последующего
    прямого fit-а к измеренным гармоникам, а не как окончательный ответ.
    """
    x = np.asarray(x0, dtype=np.complex128).copy()
    i_idx = obs["i"]
    j_idx = obs["j"]
    value = obs["value"]
    weight = obs["weight"]
    for _ in range(int(n_iters)):
        for n in range(chain_length):
            mask_left = i_idx == n
            mask_right = j_idx == n
            numer = 0.0 + 0.0j
            denom = 0.0
            if np.any(mask_left):
                xm = x[j_idx[mask_left]]
                wm = weight[mask_left]
                numer += np.sum(wm * value[mask_left] * xm)
                denom += np.sum(wm * np.abs(xm) ** 2)
            if np.any(mask_right):
                xm = x[i_idx[mask_right]]
                wm = weight[mask_right]
                numer += np.sum(wm * np.conj(value[mask_right]) * xm)
                denom += np.sum(wm * np.abs(xm) ** 2)
            if denom > 0.0:
                x[n] = numer / denom
        if np.real(x[0]) < 0.0:
            x *= -1.0
    return x


def factorization_error_by_lag(recovered_e, solved_diagonals, amplitude_floor):
    errors = []
    for p, diag in enumerate(solved_diagonals, start=1):
        amplitude = np.abs(diag)
        valid = amplitude > float(amplitude_floor)
        pred = recovered_e[:-p] * np.conj(recovered_e[p:])
        local = np.abs(pred[valid] - diag[valid])
        errors.append(float(np.mean(local)) if local.size > 0 else np.nan)
    return np.asarray(errors, dtype=np.float64)


def build_hermitian_matrix(recovered_e):
    recovered_e = np.asarray(recovered_e, dtype=np.complex128).reshape(-1)
    return recovered_e[:, None] * np.conj(recovered_e[None, :])


def canonicalize_global_phase(e_field):
    e_field = np.asarray(e_field, dtype=np.complex128).copy()
    if e_field.size == 0:
        return e_field
    if abs(e_field[0]) > 1e-12:
        e_field *= np.exp(-1j * np.angle(e_field[0]))
    if np.real(e_field[0]) < 0.0:
        e_field *= -1.0
    return e_field


def predict_harmonics_from_field(weights, e_field, lag_indices, coord_count):
    """Прямая модель: рассчитать H_p(z) из кандидатного дискретного поля E.

    Для каждого лага p и координаты z_s гармоника является valid-свёрткой
    диагонали попарных произведений E_i*conj(E_{i+p}) с ядром перекрытия
    импульса weights[k] * weights[k+p].
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    e_field = np.asarray(e_field, dtype=np.complex128).reshape(-1)
    lag_indices = np.asarray(lag_indices, dtype=np.int64).reshape(-1)
    predicted = np.empty((coord_count, lag_indices.size), dtype=np.complex128)
    pulse_count = weights.size
    for col, p in enumerate(lag_indices):
        kernel = weights[: pulse_count - p] * weights[p:]
        pair_product = e_field[:-p] * np.conj(e_field[p:])
        predicted[:, col] = np.correlate(pair_product, kernel, mode="valid")
    return predicted


def fit_field_directly_to_harmonics(
    even_weights,
    odd_weights,
    even_harmonics,
    odd_harmonics,
    lag_indices,
    initial_e,
    n_iters,
    damping,
    ridge_lambda,
):
    """Уточнить E прямым fit-ом к измеренным чётным/нечётным картам гармоник.

    Двухэтапное восстановление `H_p -> попарные произведения -> E` может
    усиливать шум, особенно для слабых лагов. Поэтому эта функция возвращается
    к исходно измеренным величинам и минимизирует расхождение между измеренными
    H_p(z) и гармониками, рассчитанными из текущего E.

    Обновление выполняется покоординатно. Если все отсчёты поля кроме E_n
    зафиксированы, каждая затронутая невязка гармоники линейна по Re(E_n) и
    Im(E_n). Поэтому каждое обновление — это маленькая вещественная МНК-задача
    с ridge-регуляризацией и damping-ом против больших нестабильных скачков.
    """
    even_weights = np.asarray(even_weights, dtype=np.float64).reshape(-1)
    odd_weights = np.asarray(odd_weights, dtype=np.float64).reshape(-1)
    even_harmonics = np.asarray(even_harmonics, dtype=np.complex128)
    odd_harmonics = np.asarray(odd_harmonics, dtype=np.complex128)
    lag_indices = np.asarray(lag_indices, dtype=np.int64).reshape(-1)
    initial_e = canonicalize_global_phase(initial_e)

    chain_length = initial_e.size
    coord_count = even_harmonics.shape[0]
    even_scale = 1.0 / np.maximum(np.sqrt(np.mean(np.abs(even_harmonics) ** 2, axis=0)), 1e-8)
    odd_scale = 1.0 / np.maximum(np.sqrt(np.mean(np.abs(odd_harmonics) ** 2, axis=0)), 1e-8)

    def residual_vector(e_field):
        # Нормируем каждый лаг перед объединением невязок. Иначе сильные
        # низкие гармоники доминируют в функционале, а слабые, но информативные
        # лаги почти не влияют на fit.
        e_field = canonicalize_global_phase(e_field)
        pred_even = predict_harmonics_from_field(even_weights, e_field, lag_indices, coord_count)
        pred_odd = predict_harmonics_from_field(odd_weights, e_field, lag_indices, coord_count)
        res_even = (pred_even - even_harmonics) * even_scale[None, :]
        res_odd = (pred_odd - odd_harmonics) * odd_scale[None, :]
        return np.concatenate(
            [
                res_even.real.ravel(),
                res_even.imag.ravel(),
                res_odd.real.ravel(),
                res_odd.imag.ravel(),
            ]
        )

    parity_payload = [
        ("even", even_weights, even_harmonics, even_scale),
        ("odd", odd_weights, odd_harmonics, odd_scale),
    ]

    recovered_e = initial_e.copy()
    for _ in range(int(n_iters)):
        pred_even = predict_harmonics_from_field(even_weights, recovered_e, lag_indices, coord_count)
        pred_odd = predict_harmonics_from_field(odd_weights, recovered_e, lag_indices, coord_count)
        prediction_map = {"even": pred_even, "odd": pred_odd}

        for n in range(chain_length):
            design_rows = []
            rhs_rows = []
            for parity_name, weights, observed, scale in parity_payload:
                predicted = prediction_map[parity_name]
                pulse_count = weights.size
                for col, p in enumerate(lag_indices):
                    kernel = weights[: pulse_count - p] * weights[p:]
                    m_count = kernel.size
                    # H_p(z_s) содержит слагаемые E_{s+m}*conj(E_{s+m+p}).
                    # E_n может быть левым членом пары (n=s+m) или правым
                    # членом пары (n=s+m+p). Кандидатные координаты ниже — это
                    # ровно те строки s, на которые влияет E_n.
                    s_left_lo = max(0, n - (m_count - 1))
                    s_left_hi = min(coord_count - 1, n)
                    s_right_lo = max(0, n - p - (m_count - 1))
                    s_right_hi = min(coord_count - 1, n - p)
                    candidate_s = set()
                    if s_left_hi >= s_left_lo:
                        candidate_s.update(range(s_left_lo, s_left_hi + 1))
                    if s_right_hi >= s_right_lo:
                        candidate_s.update(range(s_right_lo, s_right_hi + 1))

                    for s in candidate_s:
                        alpha = 0.0 + 0.0j
                        beta = 0.0 + 0.0j
                        current_term = 0.0 + 0.0j

                        m_left = n - s
                        if 0 <= m_left < m_count and (n + p) < chain_length:
                            alpha = kernel[m_left] * np.conj(recovered_e[n + p])
                            current_term += alpha * recovered_e[n]

                        m_right = n - s - p
                        if 0 <= m_right < m_count and (n - p) >= 0:
                            beta = kernel[m_right] * recovered_e[n - p]
                            current_term += beta * np.conj(recovered_e[n])

                        if alpha == 0.0 and beta == 0.0:
                            continue

                        # Убираем старый вклад E_n из текущего предсказания,
                        # затем ищем такое E_n, которое лучше всего объясняет
                        # оставшуюся цель. Комплексное выражение
                        # alpha*E_n + beta*conj(E_n) линейно по Re(E_n) и
                        # Im(E_n), что даёт два вещественных столбца ниже.
                        target = observed[s, col] - (predicted[s, col] - current_term)
                        y = target * scale[col]
                        c1 = (alpha + beta) * scale[col]
                        c2 = 1j * (alpha - beta) * scale[col]
                        design_rows.append([np.real(c1), np.real(c2)])
                        rhs_rows.append(np.real(y))
                        design_rows.append([np.imag(c1), np.imag(c2)])
                        rhs_rows.append(np.imag(y))

            if len(design_rows) < 2:
                continue

            design = np.asarray(design_rows, dtype=np.float64)
            rhs = np.asarray(rhs_rows, dtype=np.float64)
            ridge = np.sqrt(float(ridge_lambda)) * np.eye(2, dtype=np.float64)
            augmented_design = np.vstack([design, ridge])
            augmented_rhs = np.concatenate([rhs, ridge @ np.array([np.real(recovered_e[n]), np.imag(recovered_e[n])])])
            solution, *_ = np.linalg.lstsq(augmented_design, augmented_rhs, rcond=None)
            updated = solution[0] + 1j * solution[1]
            recovered_e[n] = (1.0 - float(damping)) * recovered_e[n] + float(damping) * updated

        recovered_e = canonicalize_global_phase(recovered_e)
        pred_even = predict_harmonics_from_field(even_weights, recovered_e, lag_indices, coord_count)
        pred_odd = predict_harmonics_from_field(odd_weights, recovered_e, lag_indices, coord_count)
        pred_stack = np.concatenate([pred_even.ravel(), pred_odd.ravel()])
        obs_stack = np.concatenate([even_harmonics.ravel(), odd_harmonics.ravel()])
        denom = np.vdot(pred_stack, pred_stack).real
        if denom > 1e-12:
            scale2 = max(np.vdot(pred_stack, obs_stack).real / denom, 1e-12)
            recovered_e *= np.sqrt(scale2)
        recovered_e = canonicalize_global_phase(recovered_e)

    pred_even = predict_harmonics_from_field(even_weights, recovered_e, lag_indices, coord_count)
    pred_odd = predict_harmonics_from_field(odd_weights, recovered_e, lag_indices, coord_count)
    return {
        "success": True,
        "message": "Прямой ALS-fit завершён",
        "n_iters": int(n_iters),
        "cost": float(np.mean(np.abs(residual_vector(recovered_e)) ** 2)),
        "field": recovered_e,
        "modeled_even": pred_even,
        "modeled_odd": pred_odd,
        "residual_vector": residual_vector(recovered_e),
    }


def save_matlab_bundle(output_dir, stem, suffix_tag, payload):
    output_dir = Path(output_dir)
    mat_path = output_dir / f"{stem}_{suffix_tag}_matlab_data.mat"
    script_stem = matlab_safe_stem(f"open_{stem}_{suffix_tag}_in_matlab")
    script_path = output_dir / f"{script_stem}.m"
    savemat(mat_path, payload)
    script_text = f"""this_dir = fileparts(mfilename('fullpath'));
data = load(fullfile(this_dir, '{mat_path.name}'));

f1 = figure('Color', 'w', 'Name', 'Восстановленные комплексные амплитуды');
subplot(2,1,1);
plot(data.chain_distance_m, data.E_amplitude, 'LineWidth', 1.4);
grid on; xlabel('Расстояние (m)'); ylabel('|E|'); title('Модуль восстановленной комплексной амплитуды');
subplot(2,1,2);
plot(data.chain_distance_m, data.E_phase, 'LineWidth', 1.4);
grid on; xlabel('Расстояние (m)'); ylabel('arg(E) (rad)'); title('Фаза восстановленной комплексной амплитуды');

f2 = figure('Color', 'w', 'Name', 'Восстановленные попарные произведения');
subplot(2,1,1);
imagesc(data.lag_indices, data.chain_distance_m, data.reconstructed_product_phase);
axis xy; colorbar; xlabel('Лаг p'); ylabel('Расстояние (m)'); title('arg(E_i conj(E_{{i+p}}))');
subplot(2,1,2);
imagesc(data.lag_indices, data.chain_distance_m, data.reconstructed_product_amplitude);
axis xy; colorbar; xlabel('Лаг p'); ylabel('Расстояние (m)'); title('|E_i conj(E_{{i+p}})|');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return {"mat": mat_path, "script": script_path}


def main():
    parser = argparse.ArgumentParser(
        description="Восстановить комплексное дискретное поле E_i по одному свипу H_p(z) через факторизацию X = E E^H."
    )
    parser.add_argument("dat_path", help="Путь к .dat-файлу")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Каталог для выходных файлов")
    parser.add_argument("--scan-rate", type=float, default=None, help="Необязательная частота записи рефлектограмм в Hz")
    parser.add_argument("--fiber-z-min", type=float, default=105.0, help="Начало полезного участка волокна в метрах")
    parser.add_argument("--fiber-z-max", type=float, default=280.0, help="Конец полезного участка волокна в метрах")
    parser.add_argument("--pulse-z-min", type=float, default=75.0, help="Начало поддержки импульса в метрах")
    parser.add_argument("--pulse-z-max", type=float, default=85.0, help="Конец поддержки импульса в метрах")
    parser.add_argument("--zero-level-z", type=float, default=70.0, help="Координата нулевого уровня для весов импульса")
    parser.add_argument("--lambda0-nm", type=float, default=1550.0, help="Центральная длина волны в nm")
    parser.add_argument("--sweep-span-pm", type=float, default=3.125, help="Размах одного свипа длины волны в pm")
    parser.add_argument("--rolling-window", type=int, default=64, help="Окно сглаживания для детектора сбросов, в трассах")
    parser.add_argument("--min-period-s", type=float, default=0.05, help="Минимальный период свипа для детектора сбросов")
    parser.add_argument("--max-period-s", type=float, default=None, help="Максимальный период свипа для детектора сбросов")
    parser.add_argument("--prominence-sigma", type=float, default=2.0, help="Порог детектора сбросов в робастных sigma")
    parser.add_argument("--refine-window-fraction", type=float, default=0.15, help="Окно локального уточнения как доля найденного периода")
    parser.add_argument("--reset-time-shift-ms", type=float, default=3.0, help="Сдвинуть найденные времена сбросов позже на это число ms")
    parser.add_argument("--reset-period-override-ms", type=float, default=None, help="Использовать этот фиксированный период сбросов в ms")
    parser.add_argument("--reset-anchor-time-s", type=float, default=None, help="Опорное время для фиксированной сетки сбросов")
    parser.add_argument("--shared-reset-detection", action="store_true", help="Использовать общую сетку сбросов для чётных и нечётных трасс")
    parser.add_argument("--max-reset-time-s", type=float, default=None, help="Оставить только сбросы не позже этого времени")
    parser.add_argument("--reset-detection-end-time-s", type=float, default=None, help="Использовать только трассы до этого времени при поиске сбросов")
    parser.add_argument("--regularize-reset-grid", action="store_true", help="Спроецировать найденные сбросы на регулярную сетку с найденным периодом")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Вычесть базовый уровень, оцененный по последним N метрам каждой трассы")
    parser.add_argument("--ridge-lambda", type=float, default=1e-6, help="Ridge-регуляризация для линейной МНК-задачи по диагоналям")
    parser.add_argument("--amplitude-floor", type=float, default=1e-4, help="Игнорировать решённые попарные произведения ниже этого порога")
    parser.add_argument("--als-iters", type=int, default=20, help="Число итераций попеременной минимизации для восстановления E")
    parser.add_argument("--sweep-index", type=int, default=0, help="Индекс свипа с нуля; отрицательный индекс отсчитывается с конца")
    parser.add_argument("--lag-min", type=int, default=1, help="Минимальный лаг p для прямого нелинейного fit-а")
    parser.add_argument("--lag-max", type=int, default=None, help="Максимальный лаг p для прямого нелинейного fit-а")
    parser.add_argument("--direct-iters", type=int, default=20, help="Число ALS-итераций прямого fit-а к измеренным гармоникам")
    parser.add_argument("--direct-damping", type=float, default=0.25, help="Коэффициент damping-а для прямых ALS-обновлений")
    parser.add_argument("--direct-ridge-lambda", type=float, default=1e-3, help="Ridge-регуляризация для каждого скалярного ALS-обновления")
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
    data, _, _, baseline_start_m, baseline_end_m = subtract_trace_baseline_from_tail(
        data,
        distance_axis_m,
        args.baseline_tail_m,
    )

    fiber_mask = (distance_axis_m >= float(args.fiber_z_min)) & (distance_axis_m <= float(args.fiber_z_max))
    if not np.any(fiber_mask):
        raise ValueError("Окно полезного волокна пустое")
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
    lag_distances_m = lag_indices.astype(np.float64) * float(distance_step_m)
    lambda0_m = float(args.lambda0_nm) * 1e-9
    sweep_span_m = float(args.sweep_span_pm) * 1e-12
    delta_beta_span = -2.0 * np.pi * float(n_eff) * sweep_span_m / (lambda0_m**2)

    selected_sweep_harmonics = {}
    dominant_periods = {}
    reset_counts = {}
    complete_sweep_counts = {}
    selected_sweep_index = None
    shared_reset_times_s = None
    shared_dominant_period_s = None
    if args.shared_reset_detection:
        shared_global_indices = np.arange(data.shape[0], dtype=np.int64)
        shared_data = data[:, fiber_mask]
        if args.reset_detection_end_time_s is not None:
            keep = shared_global_indices.astype(np.float64) / float(result["scan_rate"]) <= float(args.reset_detection_end_time_s)
            shared_global_indices = shared_global_indices[keep]
            shared_data = shared_data[keep]
        shared_reset_times_s, shared_dominant_period_s = detect_reset_times(
            shared_data,
            shared_global_indices,
            result["scan_rate"],
            rolling_window=args.rolling_window,
            min_period_s=args.min_period_s,
            prominence_sigma=args.prominence_sigma,
            refine_window_fraction=args.refine_window_fraction,
            max_period_s=args.max_period_s,
        )
        shared_reset_times_s = shared_reset_times_s + 1e-3 * float(args.reset_time_shift_ms)
        if args.max_reset_time_s is not None:
            shared_reset_times_s = shared_reset_times_s[shared_reset_times_s <= float(args.max_reset_time_s)]
        if args.regularize_reset_grid:
            shared_reset_times_s = regularize_reset_grid(shared_reset_times_s, shared_dominant_period_s)
        if args.reset_period_override_ms is not None:
            override_period_s = 1e-3 * float(args.reset_period_override_ms)
            if args.reset_anchor_time_s is None:
                if shared_reset_times_s.size == 0:
                    raise ValueError("Нельзя определить опорный сброс по пустому списку сбросов")
                anchor_time_s = float(shared_reset_times_s[0])
            else:
                anchor_time_s = float(args.reset_anchor_time_s)
            max_grid_time_s = float(args.max_reset_time_s) if args.max_reset_time_s is not None else float(shared_global_indices[-1]) / float(result["scan_rate"])
            shared_reset_times_s = build_periodic_reset_grid(
                anchor_time_s=anchor_time_s,
                period_s=override_period_s,
                min_time_s=0.0,
                max_time_s=max_grid_time_s,
            )
            shared_dominant_period_s = override_period_s

    for parity in ["even", "odd"]:
        fiber_data = data[:, fiber_mask][0::2] if parity == "even" else data[:, fiber_mask][1::2]
        parity_global_indices = (
            np.arange(0, result["refls_count"], 2, dtype=np.int64)
            if parity == "even"
            else np.arange(1, result["refls_count"], 2, dtype=np.int64)
        )
        parity_time_s = parity_global_indices.astype(np.float64) / float(result["scan_rate"])
        if args.shared_reset_detection:
            reset_times_s = shared_reset_times_s.copy()
            dominant_period_s = float(shared_dominant_period_s)
        else:
            detection_indices = parity_global_indices
            detection_data = fiber_data
            if args.reset_detection_end_time_s is not None:
                keep = detection_indices.astype(np.float64) / float(result["scan_rate"]) <= float(args.reset_detection_end_time_s)
                detection_indices = detection_indices[keep]
                detection_data = detection_data[keep]
            reset_times_s, dominant_period_s = detect_reset_times(
                detection_data,
                detection_indices,
                result["scan_rate"],
                rolling_window=args.rolling_window,
                min_period_s=args.min_period_s,
                prominence_sigma=args.prominence_sigma,
                refine_window_fraction=args.refine_window_fraction,
                max_period_s=args.max_period_s,
            )
            reset_times_s = reset_times_s + 1e-3 * float(args.reset_time_shift_ms)
            if args.max_reset_time_s is not None:
                reset_times_s = reset_times_s[reset_times_s <= float(args.max_reset_time_s)]
            if args.regularize_reset_grid:
                reset_times_s = regularize_reset_grid(reset_times_s, dominant_period_s)
            if args.reset_period_override_ms is not None:
                override_period_s = 1e-3 * float(args.reset_period_override_ms)
                if args.reset_anchor_time_s is None:
                    if reset_times_s.size == 0:
                        raise ValueError("Нельзя определить опорный сброс по пустому списку сбросов")
                    anchor_time_s = float(reset_times_s[0])
                else:
                    anchor_time_s = float(args.reset_anchor_time_s)
                max_grid_time_s = float(args.max_reset_time_s) if args.max_reset_time_s is not None else float(detection_indices[-1]) / float(result["scan_rate"])
                reset_times_s = build_periodic_reset_grid(
                    anchor_time_s=anchor_time_s,
                    period_s=override_period_s,
                    min_time_s=0.0,
                    max_time_s=max_grid_time_s,
                )
                dominant_period_s = override_period_s
        sweep_intervals_s = build_sweep_intervals(reset_times_s)
        harmonic_cube, _, _ = harmonics_for_sweeps(
            fiber_data,
            parity_time_s,
            sweep_intervals_s,
            delta_beta_span=delta_beta_span,
            lag_distances_m=lag_distances_m,
        )
        requested_sweep_index = int(args.sweep_index)
        resolved_sweep_index = requested_sweep_index if requested_sweep_index >= 0 else harmonic_cube.shape[0] + requested_sweep_index
        if not (0 <= resolved_sweep_index < harmonic_cube.shape[0]):
            raise ValueError(
                f"sweep_index {args.sweep_index} вне диапазона для parity '{parity}', полных свипов: {harmonic_cube.shape[0]}"
            )
        selected_sweep_harmonics[parity] = harmonic_cube[resolved_sweep_index]
        dominant_periods[parity] = dominant_period_s
        reset_counts[parity] = reset_times_s.size
        complete_sweep_counts[parity] = harmonic_cube.shape[0]
        selected_sweep_index = resolved_sweep_index

    solved = solve_diagonal_entries(
        selected_sweep_harmonics["even"],
        selected_sweep_harmonics["odd"],
        even_weights,
        odd_weights,
        ridge_lambda=args.ridge_lambda,
    )
    obs = build_observation_lists(solved["solved_diagonals"], args.amplitude_floor)
    magnitudes0 = solve_log_magnitudes(obs, obs["chain_length"], ridge_lambda=args.ridge_lambda)
    phases0 = solve_recursive_phases(obs, obs["chain_length"])
    x0 = magnitudes0 * np.exp(1j * phases0)
    initial_e = alternating_rank1_hermitian(obs, obs["chain_length"], x0=x0, n_iters=args.als_iters)

    lag_max = int(args.lag_max) if args.lag_max is not None else int(lag_indices[-1])
    lag_mask = (lag_indices >= int(args.lag_min)) & (lag_indices <= lag_max)
    if not np.any(lag_mask):
        raise ValueError("Выбранное окно лагов пустое")
    fit_lag_indices = lag_indices[lag_mask]
    direct_fit = fit_field_directly_to_harmonics(
        even_weights=even_weights,
        odd_weights=odd_weights,
        even_harmonics=selected_sweep_harmonics["even"][:, lag_mask],
        odd_harmonics=selected_sweep_harmonics["odd"][:, lag_mask],
        lag_indices=fit_lag_indices,
        initial_e=initial_e,
        n_iters=args.direct_iters,
        damping=args.direct_damping,
        ridge_lambda=args.direct_ridge_lambda,
    )
    recovered_e = direct_fit["field"]

    reconstructed_products = build_hermitian_matrix(recovered_e)
    reconstructed_diagonals = np.full_like(solved["diagonal_matrix"], np.nan + 0j)
    for p in range(1, pulse_count):
        diag = recovered_e[:-p] * np.conj(recovered_e[p:])
        reconstructed_diagonals[: diag.size, p - 1] = diag
    factorization_error = factorization_error_by_lag(
        recovered_e,
        solved["solved_diagonals"],
        amplitude_floor=args.amplitude_floor,
    )

    chain_distance_m = fiber_distance_m[0] + np.arange(obs["chain_length"], dtype=np.float64) * float(distance_step_m)
    e_amplitude = np.abs(recovered_e)
    e_phase = np.unwrap(np.angle(recovered_e))
    reconstructed_product_amplitude = np.abs(reconstructed_diagonals)
    reconstructed_product_phase = np.angle(reconstructed_diagonals)

    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes1[0].plot(chain_distance_m, e_amplitude, color="#1F77B4", linewidth=1.5)
    axes1[0].set_ylabel("|E|")
    axes1[0].set_title("Восстановленные комплексные амплитуды дискретов")
    axes1[0].grid(alpha=0.25)
    axes1[1].plot(chain_distance_m, e_phase, color="#111111", linewidth=1.4)
    axes1[1].set_xlabel("Расстояние (m)")
    axes1[1].set_ylabel("arg(E) (rad)")
    axes1[1].grid(alpha=0.25)
    e_png_path = output_dir / f"{dat_path.stem}_recovered_complex_amplitudes.png"
    fig1.savefig(e_png_path, dpi=200)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    im0 = axes2[0].imshow(
        reconstructed_product_phase,
        aspect="auto",
        origin="lower",
        cmap="twilight",
        extent=[lag_indices[0], lag_indices[-1], chain_distance_m[0], chain_distance_m[-1]],
    )
    axes2[0].set_ylabel("Расстояние (m)")
    axes2[0].set_title("Восстановленные попарные произведения: arg(E_i conj(E_{i+p}))")
    fig2.colorbar(im0, ax=axes2[0], label="Фаза (rad)")
    im1 = axes2[1].imshow(
        reconstructed_product_amplitude,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[lag_indices[0], lag_indices[-1], chain_distance_m[0], chain_distance_m[-1]],
    )
    axes2[1].set_xlabel("Лаг p")
    axes2[1].set_ylabel("Расстояние (m)")
    axes2[1].set_title("Восстановленные попарные произведения: |E_i conj(E_{i+p})|")
    fig2.colorbar(im1, ax=axes2[1], label="Амплитуда")
    products_png_path = output_dir / f"{dat_path.stem}_recovered_pair_products.png"
    fig2.savefig(products_png_path, dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax3.plot(lag_indices, solved["residual_rms"], linewidth=1.5, label="RMS линейного решения диагоналей")
    ax3.plot(lag_indices, factorization_error, linewidth=1.5, label="Невязка rank-1 факторизации")
    ax3.axvspan(lag_indices[0], int(args.lag_min), color="#DDDDDD", alpha=0.35, linewidth=0)
    if lag_max < int(lag_indices[-1]):
        ax3.axvspan(lag_max, lag_indices[-1], color="#DDDDDD", alpha=0.35, linewidth=0)
    ax3.set_xlabel("Лаг p")
    ax3.set_ylabel("Ошибка")
    ax3.set_title("Невязка модели по лагам")
    ax3.grid(alpha=0.25)
    ax3.legend()
    error_png_path = output_dir / f"{dat_path.stem}_complex_amplitude_model_error_by_lag.png"
    fig3.savefig(error_png_path, dpi=200)
    plt.close(fig3)

    matlab_saved_paths = save_matlab_bundle(
        output_dir=output_dir,
        stem=dat_path.stem,
        suffix_tag="complex_amplitude_factorization_single_sweep",
        payload={
            "chain_distance_m": chain_distance_m[:, None],
            "lag_indices": lag_indices[:, None],
            "lag_distances_m": lag_distances_m[:, None],
            "pulse_distance_m": pulse_distance_m[:, None],
            "even_weights": even_weights[:, None],
            "odd_weights": odd_weights[:, None],
            "E": recovered_e[:, None],
            "E_amplitude": e_amplitude[:, None],
            "E_phase": e_phase[:, None],
            "reconstructed_products": reconstructed_products,
            "reconstructed_product_amplitude": reconstructed_product_amplitude,
            "reconstructed_product_phase": reconstructed_product_phase,
            "solved_diagonal_matrix": solved["diagonal_matrix"],
            "solved_diagonal_amplitude": np.abs(solved["diagonal_matrix"]),
            "solved_diagonal_phase": np.angle(solved["diagonal_matrix"]),
            "factorization_error_by_lag": factorization_error[:, None],
            "linear_residual_rms": solved["residual_rms"][:, None],
            "even_selected_sweep_harmonics": selected_sweep_harmonics["even"],
            "odd_selected_sweep_harmonics": selected_sweep_harmonics["odd"],
            "even_direct_modeled_harmonics": direct_fit["modeled_even"],
            "odd_direct_modeled_harmonics": direct_fit["modeled_odd"],
            "sweep_index": np.array([[int(selected_sweep_index)]], dtype=np.int32),
            "fit_lag_indices": fit_lag_indices[:, None],
            "distance_step_m": np.array([[distance_step_m]], dtype=np.float64),
            "baseline_window_start_m": np.array([[baseline_start_m]], dtype=np.float64),
            "baseline_window_end_m": np.array([[baseline_end_m]], dtype=np.float64),
            "zero_level_actual_m": np.array([[zero_level_actual_m]], dtype=np.float64),
            "zero_level_index": np.array([[zero_index]], dtype=np.int32),
            "shared_reset_detection": np.array([[int(bool(args.shared_reset_detection))]], dtype=np.int32),
            "regularize_reset_grid": np.array([[int(bool(args.regularize_reset_grid))]], dtype=np.int32),
            "reset_period_override_ms": np.array(
                [[np.nan if args.reset_period_override_ms is None else float(args.reset_period_override_ms)]],
                dtype=np.float64,
            ),
            "reset_anchor_time_s": np.array(
                [[np.nan if args.reset_anchor_time_s is None else float(args.reset_anchor_time_s)]],
                dtype=np.float64,
            ),
        },
    )

    print(f"file: {dat_path}")
    print(f"scan_rate_hz: {result['scan_rate']}")
    print(f"fiber_distance_start_m: {fiber_distance_m[0]:.6f}")
    print(f"fiber_distance_end_m: {fiber_distance_m[-1]:.6f}")
    print(f"baseline_tail_m: {args.baseline_tail_m}")
    print(f"baseline_window_start_m: {baseline_start_m:.6f}")
    print(f"baseline_window_end_m: {baseline_end_m:.6f}")
    print(f"reset_time_shift_ms: {args.reset_time_shift_ms}")
    if args.reset_period_override_ms is not None:
        print(f"reset_period_override_ms: {args.reset_period_override_ms}")
    if args.reset_anchor_time_s is not None:
        print(f"reset_anchor_time_s: {args.reset_anchor_time_s}")
    print(f"shared_reset_detection: {args.shared_reset_detection}")
    print(f"regularize_reset_grid: {args.regularize_reset_grid}")
    if args.max_period_s is not None:
        print(f"max_period_s: {args.max_period_s}")
    if args.max_reset_time_s is not None:
        print(f"max_reset_time_s: {args.max_reset_time_s}")
    if args.reset_detection_end_time_s is not None:
        print(f"reset_detection_end_time_s: {args.reset_detection_end_time_s}")
    print(f"pulse_discrete_count_N: {pulse_count}")
    print(f"chain_length: {obs['chain_length']}")
    print(f"sweep_index_requested: {args.sweep_index}")
    print(f"sweep_index_selected: {selected_sweep_index}")
    print(f"lag_min: {args.lag_min}")
    print(f"lag_max: {lag_max}")
    print(f"fit_lag_count: {fit_lag_indices.size}")
    print(f"direct_damping: {args.direct_damping}")
    print(f"direct_ridge_lambda: {args.direct_ridge_lambda}")
    print(f"sweep_span_pm: {args.sweep_span_pm}")
    print(f"delta_beta_span: {delta_beta_span:.10e}")
    for parity in ["even", "odd"]:
        print(f"{parity}_dominant_period_s: {dominant_periods[parity]:.10f}")
        print(f"{parity}_reset_count: {reset_counts[parity]}")
        print(f"{parity}_complete_sweeps: {complete_sweep_counts[parity]}")
    print(f"linear_residual_rms_mean: {np.nanmean(solved['residual_rms']):.10e}")
    print(f"factorization_error_mean: {np.nanmean(factorization_error):.10e}")
    print(f"direct_fit_success: {direct_fit['success']}")
    print(f"direct_fit_iters: {direct_fit['n_iters']}")
    print(f"direct_fit_cost: {direct_fit['cost']:.10e}")
    print(f"direct_fit_residual_rms: {np.sqrt(np.mean(direct_fit['residual_vector'] ** 2)):.10e}")
    print(f"E_amplitude_mean: {np.mean(e_amplitude):.10e}")
    print(f"E_amplitude_max: {np.max(e_amplitude):.10e}")
    print(f"complex_amplitude_png_saved_to: {e_png_path}")
    print(f"pair_products_png_saved_to: {products_png_path}")
    print(f"error_by_lag_png_saved_to: {error_png_path}")
    print(f"matlab_data_saved_to: {matlab_saved_paths['mat']}")
    print(f"matlab_open_script_saved_to: {matlab_saved_paths['script']}")


if __name__ == "__main__":
    main()
