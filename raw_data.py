#Различные модули для обработки сырых данных с фотодиода
import matplotlib.pyplot as plt
import numpy as np
import struct
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert
from scipy.optimize import curve_fit, least_squares
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Обновленная функция чтения данных (для полноты)
def read_reflectograms(filename, scan_rate=None, subtract_dark=True):
    with open(filename, 'rb') as f:
        hdr_ver = struct.unpack('<i', f.read(4))[0]
        time_creation = struct.unpack('<q', f.read(8))[0]
        segment_size = struct.unpack('<i', f.read(4))[0]
        refls_count = struct.unpack('<i', f.read(4))[0]
        sampling_rate = struct.unpack('<i', f.read(4))[0]
        stored_scan_rate = struct.unpack('<i', f.read(4))[0]
        if scan_rate is None:
            scan_rate = stored_scan_rate
        real_segment_size = segment_size
        if hdr_ver > 2:
            x1 = struct.unpack('<i', f.read(4))[0]
            x2 = struct.unpack('<i', f.read(4))[0]
            real_segment_size = x2 - x1

        tau = None
        pump_cur = None
        if hdr_ver == 4:
            tau = struct.unpack('<ii', f.read(8))
            pump_cur = struct.unpack('<i', f.read(4))[0]

        # Чтение и преобразование
        data_size = refls_count * real_segment_size
        buffer = f.read(data_size * 2)
        data = np.frombuffer(buffer, dtype=np.int16)
        data = data.reshape(refls_count, real_segment_size)
        data = data.astype(np.float32) / 8192.0

        # === Удаление темнового сигнала ===
        if subtract_dark:
            # 1. Найдём максимум сигнала по оси времени
            avg_trace = np.mean(data, axis=0)
            max_idx = np.argmax(avg_trace)

            # 2. Переходим на 10% дальше по времени
            offset_idx = int(min(real_segment_size - 1, max_idx * 2))

            # 3. Усреднение по всем трассам в этой точке (или окне ±3)
            window = 3000000
            idx_range = slice(max(0, offset_idx), min(real_segment_size, offset_idx + window + 1))
            dark_level = 0#np.mean(data[:, idx_range])

            # 4. Вычитаем
            data -= dark_level
            print(f"🔧 Темновой сигнал (offset) удалён: среднее = {dark_level:.5f}")

        return {
            'data': data,
            'hdr_ver': hdr_ver,
            'time_creation': time_creation,
            'sampling_rate': sampling_rate,
            'scan_rate': scan_rate,
            'stored_scan_rate': stored_scan_rate,
            'segment_size': segment_size,
            'real_segment_size': real_segment_size,
            'refls_count': refls_count,
            'tau': tau,
            'pump_cur': pump_cur
        }


def read_reflectometer_bin(filename):
    """
    Читает bin-файл калибровки/скана рефлектометра формата CRCAP1.

    Текущая реализация использует размеры из заголовка и вычисляет offset
    как остаток между полным размером файла и размером float32-полезной нагрузки.
    Данные возвращаются как memmap, чтобы не загружать гигабайтный файл в RAM.
    """
    import os

    with open(filename, "rb") as f:
        magic = f.read(8)
        if magic[:6] != b"CRCAP1":
            raise ValueError("Unsupported bin file magic")

        version = struct.unpack("<I", f.read(4))[0]
        x_count = struct.unpack("<I", f.read(4))[0]
        trace_count = struct.unpack("<I", f.read(4))[0]
        x_step = struct.unpack("<f", f.read(4))[0]
        x_start = struct.unpack("<f", f.read(4))[0]

    file_size = os.path.getsize(filename)
    payload_size = int(x_count) * int(trace_count) * 4
    data_offset = file_size - payload_size
    if data_offset < 0:
        raise ValueError("File is smaller than declared payload")

    data = np.memmap(
        filename,
        dtype="<f4",
        mode="r",
        offset=data_offset,
        shape=(trace_count, x_count),
    )

    x_axis = x_start + np.arange(x_count, dtype=np.float32) * x_step

    return {
        "data": data,
        "magic": magic,
        "version": version,
        "x_count": x_count,
        "trace_count": trace_count,
        "x_step": x_step,
        "x_start": x_start,
        "x_axis": x_axis,
        "data_offset": data_offset,
        "file_size": file_size,
    }


def interpolate_sweep_read_value_per_reflectogram(
    dat_result,
    sweep_csv_path,
    pm_per_unit=0.018,
    use_column="read_value",
):
    """
    Интерполирует значение sweep.csv на каждый кадр рефлектограммы.

    Возвращает
    ----------
    time_axis_s : ndarray
        Время каждого кадра в секундах.
    sweep_units : ndarray
        Интерполированное значение регулятора/датчика в units.
    wavelength_shift_pm : ndarray
        То же в pm, считая pm_per_unit.
    sweep_df : DataFrame
        Исходный sweep.csv.
    """
    sweep_df = pd.read_csv(sweep_csv_path)
    if "time_ms" not in sweep_df.columns:
        raise ValueError("sweep csv must contain time_ms")
    if use_column not in sweep_df.columns:
        raise ValueError(f"sweep csv must contain {use_column}")

    n_frames = int(dat_result["refls_count"])
    scan_rate = float(dat_result["scan_rate"])
    time_axis_s = np.arange(n_frames, dtype=np.float64) / scan_rate

    sweep_time_s = sweep_df["time_ms"].to_numpy(dtype=np.float64) / 1000.0
    sweep_units_src = sweep_df[use_column].to_numpy(dtype=np.float64)

    sweep_units = np.interp(
        time_axis_s,
        sweep_time_s,
        sweep_units_src,
        left=sweep_units_src[0],
        right=sweep_units_src[-1],
    )
    wavelength_shift_pm = sweep_units * pm_per_unit

    return time_axis_s, sweep_units, wavelength_shift_pm, sweep_df


def fit_real_harmonics_vs_delta_beta(signal_matrix, delta_beta, L_values_m, include_constant=True):
    """
    Для каждого столбца signal_matrix строит МНК-разложение по базису
        1, cos(2*delta_beta*L), sin(2*delta_beta*L)
    для набора L_values_m.

    Параметры
    ---------
    signal_matrix : ndarray, shape (n_samples, n_coords)
    delta_beta : ndarray, shape (n_samples,)
        Изменение волнового числа в среде [1/м].
    L_values_m : ndarray, shape (n_L,)
        Длины/разносы дискретов в метрах.

    Возвращает
    ----------
    result : dict
        constant, cos_coeffs, sin_coeffs, amplitude, phase, design_matrix
    """
    Y = np.asarray(signal_matrix, dtype=np.float64)
    delta_beta = np.asarray(delta_beta, dtype=np.float64).reshape(-1)
    L_values_m = np.asarray(L_values_m, dtype=np.float64).reshape(-1)

    if Y.shape[0] != delta_beta.shape[0]:
        raise ValueError("signal_matrix and delta_beta must have matching sample axis")

    cols = []
    if include_constant:
        cols.append(np.ones_like(delta_beta))
    phase_arg = 2.0 * delta_beta[:, None] * L_values_m[None, :]
    cols.extend([np.cos(phase_arg[:, i]) for i in range(phase_arg.shape[1])])
    cols.extend([np.sin(phase_arg[:, i]) for i in range(phase_arg.shape[1])])
    X = np.column_stack(cols)

    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    offset = 1 if include_constant else 0
    constant = coeffs[0] if include_constant else None
    cos_coeffs = coeffs[offset : offset + L_values_m.size]
    sin_coeffs = coeffs[offset + L_values_m.size : offset + 2 * L_values_m.size]
    amplitude = np.sqrt(cos_coeffs**2 + sin_coeffs**2)
    phase = np.arctan2(sin_coeffs, cos_coeffs)

    return {
        "constant": constant,
        "cos_coeffs": cos_coeffs,
        "sin_coeffs": sin_coeffs,
        "amplitude": amplitude,
        "phase": phase,
        "design_matrix": X,
    }


def extract_even_odd_pulse_weights(
    data,
    z_axis_m,
    pulse_z_min=50.0,
    pulse_z_max=75.0,
    baseline_z=40.0,
    threshold_fraction=0.01,
):
    """
    Строит средние формы импульса для чётных и нечётных рефлектограмм
    и оставляет только общий support, где хотя бы одна форма выше порога.
    """
    z_axis_m = np.asarray(z_axis_m, dtype=np.float64)
    pulse_mask = (z_axis_m >= pulse_z_min) & (z_axis_m <= pulse_z_max)
    if not np.any(pulse_mask):
        raise ValueError("Pulse window is empty")

    pulse_data = np.asarray(data[:, pulse_mask], dtype=np.float64)
    baseline_idx = int(np.argmin(np.abs(z_axis_m - baseline_z)))
    baseline = np.asarray(data[:, baseline_idx], dtype=np.float64)[:, None]
    pulse_data = pulse_data - baseline

    even_mean = pulse_data[0::2].mean(axis=0)
    odd_mean = pulse_data[1::2].mean(axis=0)
    pulse_z = z_axis_m[pulse_mask]

    max_height = float(max(np.max(even_mean), np.max(odd_mean)))
    threshold = threshold_fraction * max_height
    support_mask = (even_mean > threshold) | (odd_mean > threshold)
    if not np.any(support_mask):
        raise ValueError("No pulse support above threshold")

    return {
        "z": pulse_z[support_mask],
        "even_weights": even_mean[support_mask],
        "odd_weights": odd_mean[support_mask],
        "full_z": pulse_z,
        "full_even": even_mean,
        "full_odd": odd_mean,
        "threshold": threshold,
    }


def real_fit_to_complex_harmonics(fit_result):
    """
    Преобразует коэффициенты разложения
        c_p cos(theta_p) + s_p sin(theta_p)
    в комплексные гармоники H_p, такие что
        2 Re(H_p exp(i theta_p)) = c_p cos(theta_p) + s_p sin(theta_p).
    Тогда H_p = 0.5 * (c_p - i s_p).
    """
    cos_coeffs = np.asarray(fit_result["cos_coeffs"], dtype=np.float64)
    sin_coeffs = np.asarray(fit_result["sin_coeffs"], dtype=np.float64)
    return 0.5 * (cos_coeffs - 1j * sin_coeffs)


def model_lag_harmonics_from_phases(weights, phases, amplitude_scale=1.0, max_lag=None):
    """
    Вычисляет модельные комплексные гармоники по лагам p = 1..max_lag:
        H_p = a0^2 * sum_m A_m A_{m+p} exp(i(phi_m - phi_{m+p})).
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    if weights.size != phases.size:
        raise ValueError("weights and phases must have the same length")

    n = weights.size
    if max_lag is None:
        max_lag = n - 1
    max_lag = int(min(max_lag, n - 1))
    harmonics = np.zeros(max_lag, dtype=np.complex128)
    scale2 = float(amplitude_scale) ** 2
    for p in range(1, max_lag + 1):
        left = weights[:-p]
        right = weights[p:]
        phase_diff = phases[:-p] - phases[p:]
        harmonics[p - 1] = scale2 * np.sum(left * right * np.exp(1j * phase_diff))
    return harmonics


def intensity_from_lag_harmonics(weights, phases, amplitude_scale, delta_beta, lag_step_m):
    """
    Собирает интенсивность из дискретной модели по лагам на сетке delta_beta.
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    delta_beta = np.asarray(delta_beta, dtype=np.float64).reshape(-1)

    max_lag = weights.size - 1
    h = model_lag_harmonics_from_phases(weights, phases, amplitude_scale, max_lag=max_lag)
    const = (float(amplitude_scale) ** 2) * np.sum(weights**2)

    signal = np.full(delta_beta.shape, const, dtype=np.float64)
    for p in range(1, max_lag + 1):
        signal += 2.0 * np.real(h[p - 1] * np.exp(1j * 2.0 * delta_beta * (p * lag_step_m)))
    return signal


def fit_shared_phases_from_harmonics(
    even_weights,
    odd_weights,
    even_harmonics,
    odd_harmonics,
    n_starts=32,
    random_seed=0,
    weights_even=None,
    weights_odd=None,
):
    """
    Нелинейный МНК для общей фазы phi_m и общего масштаба a0
    по двум наборам гармоник (even/odd).
    """
    even_weights = np.asarray(even_weights, dtype=np.float64).reshape(-1)
    odd_weights = np.asarray(odd_weights, dtype=np.float64).reshape(-1)
    even_harmonics = np.asarray(even_harmonics, dtype=np.complex128).reshape(-1)
    odd_harmonics = np.asarray(odd_harmonics, dtype=np.complex128).reshape(-1)

    if even_weights.size != odd_weights.size:
        raise ValueError("even_weights and odd_weights must have the same size")
    if even_harmonics.size != odd_harmonics.size:
        raise ValueError("even_harmonics and odd_harmonics must have the same size")

    n = even_weights.size
    max_lag = int(min(even_harmonics.size, n - 1))
    even_harmonics = even_harmonics[:max_lag]
    odd_harmonics = odd_harmonics[:max_lag]

    if weights_even is None:
        weights_even = np.ones(max_lag, dtype=np.float64)
    else:
        weights_even = np.asarray(weights_even, dtype=np.float64).reshape(-1)[:max_lag]
    if weights_odd is None:
        weights_odd = np.ones(max_lag, dtype=np.float64)
    else:
        weights_odd = np.asarray(weights_odd, dtype=np.float64).reshape(-1)[:max_lag]

    rng = np.random.default_rng(random_seed)

    def unpack_params(params):
        amplitude_scale = np.exp(params[0])
        phases = np.concatenate([[0.0], params[1:]])
        return amplitude_scale, phases

    def residual_vector(params):
        amplitude_scale, phases = unpack_params(params)
        model_even = model_lag_harmonics_from_phases(
            even_weights, phases, amplitude_scale, max_lag=max_lag
        )
        model_odd = model_lag_harmonics_from_phases(
            odd_weights, phases, amplitude_scale, max_lag=max_lag
        )

        res_even = (model_even - even_harmonics) * weights_even
        res_odd = (model_odd - odd_harmonics) * weights_odd
        return np.concatenate(
            [
                res_even.real,
                res_even.imag,
                res_odd.real,
                res_odd.imag,
            ]
        )

    amp_guess = np.sqrt(
        max(
            1e-12,
            np.max(np.abs(even_harmonics)) / max(1e-12, np.max(np.abs(even_weights)) ** 2),
            np.max(np.abs(odd_harmonics)) / max(1e-12, np.max(np.abs(odd_weights)) ** 2),
        )
    )

    solutions = []
    best = None
    for start_idx in range(int(n_starts)):
        x0 = np.zeros(n, dtype=np.float64)
        x0[0] = np.log(amp_guess)
        if start_idx > 0:
            x0[1:] = rng.uniform(-np.pi, np.pi, size=n - 1)

        result = least_squares(
            residual_vector,
            x0,
            method="trf",
            x_scale="jac",
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=100,
        )
        amplitude_scale, phases = unpack_params(result.x)
        solution = {
            "cost": float(result.cost),
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.nfev),
            "amplitude_scale": float(amplitude_scale),
            "phases": phases,
            "modeled_even": model_lag_harmonics_from_phases(
                even_weights, phases, amplitude_scale, max_lag=max_lag
            ),
            "modeled_odd": model_lag_harmonics_from_phases(
                odd_weights, phases, amplitude_scale, max_lag=max_lag
            ),
        }
        solutions.append(solution)
        if best is None or solution["cost"] < best["cost"]:
            best = solution

    solutions.sort(key=lambda item: item["cost"])
    phase_stack = np.stack([sol["phases"] for sol in solutions[: min(8, len(solutions))]], axis=0)
    phase_stability = np.std(np.unwrap(phase_stack, axis=1), axis=0)

    best["all_solutions"] = solutions
    best["phase_stability"] = phase_stability
    best["max_lag"] = max_lag
    return best


def fit_shared_phases_to_traces(
    even_weights,
    odd_weights,
    even_trace,
    odd_trace,
    even_delta_beta,
    odd_delta_beta,
    lag_step_m,
    n_starts=1,
    random_seed=0,
    initial_amplitude_scale=None,
    initial_phases=None,
):
    """
    Прямой нелинейный МНК по измеренным трассам even/odd:
        y_even(db) ~= b_even + I_even(db; a0, phi)
        y_odd(db)  ~= b_odd  + I_odd(db;  a0, phi)
    """
    even_weights = np.asarray(even_weights, dtype=np.float64).reshape(-1)
    odd_weights = np.asarray(odd_weights, dtype=np.float64).reshape(-1)
    even_trace = np.asarray(even_trace, dtype=np.float64).reshape(-1)
    odd_trace = np.asarray(odd_trace, dtype=np.float64).reshape(-1)
    even_delta_beta = np.asarray(even_delta_beta, dtype=np.float64).reshape(-1)
    odd_delta_beta = np.asarray(odd_delta_beta, dtype=np.float64).reshape(-1)

    if even_weights.size != odd_weights.size:
        raise ValueError("even_weights and odd_weights must have the same size")

    n = even_weights.size
    rng = np.random.default_rng(random_seed)

    def unpack_params(params):
        amplitude_scale = np.exp(params[0])
        phases = np.concatenate([[0.0], params[1:n]])
        even_offset = params[n]
        odd_offset = params[n + 1]
        return amplitude_scale, phases, even_offset, odd_offset

    def residual_vector(params):
        amplitude_scale, phases, even_offset, odd_offset = unpack_params(params)
        model_even = even_offset + intensity_from_lag_harmonics(
            even_weights, phases, amplitude_scale, even_delta_beta, lag_step_m
        )
        model_odd = odd_offset + intensity_from_lag_harmonics(
            odd_weights, phases, amplitude_scale, odd_delta_beta, lag_step_m
        )
        return np.concatenate([model_even - even_trace, model_odd - odd_trace])

    if initial_amplitude_scale is None:
        trace_scale = max(np.std(even_trace), np.std(odd_trace), 1e-6)
        weight_scale = max(np.sum(even_weights**2), np.sum(odd_weights**2), 1e-6)
        initial_amplitude_scale = np.sqrt(trace_scale / weight_scale)

    if initial_phases is not None:
        initial_phases = np.asarray(initial_phases, dtype=np.float64).reshape(-1)
        if initial_phases.size != n:
            raise ValueError("initial_phases must match weights size")

    solutions = []
    best = None
    for start_idx in range(int(n_starts)):
        x0 = np.zeros(n + 2, dtype=np.float64)
        x0[0] = np.log(max(initial_amplitude_scale, 1e-8))
        if initial_phases is not None and start_idx == 0:
            x0[1:n] = initial_phases[1:]
        elif start_idx > 0:
            x0[1:n] = rng.uniform(-np.pi, np.pi, size=n - 1)
        x0[n] = float(np.mean(even_trace))
        x0[n + 1] = float(np.mean(odd_trace))

        result = least_squares(
            residual_vector,
            x0,
            method="lm",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=100,
        )

        amplitude_scale, phases, even_offset, odd_offset = unpack_params(result.x)
        modeled_even = even_offset + intensity_from_lag_harmonics(
            even_weights, phases, amplitude_scale, even_delta_beta, lag_step_m
        )
        modeled_odd = odd_offset + intensity_from_lag_harmonics(
            odd_weights, phases, amplitude_scale, odd_delta_beta, lag_step_m
        )
        solution = {
            "cost": float(result.cost),
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.nfev),
            "amplitude_scale": float(amplitude_scale),
            "phases": phases,
            "even_offset": float(even_offset),
            "odd_offset": float(odd_offset),
            "modeled_even_trace": modeled_even,
            "modeled_odd_trace": modeled_odd,
        }
        solutions.append(solution)
        if best is None or solution["cost"] < best["cost"]:
            best = solution

    solutions.sort(key=lambda item: item["cost"])
    phase_stack = np.stack([sol["phases"] for sol in solutions[: min(8, len(solutions))]], axis=0)
    phase_stability = np.std(np.unwrap(phase_stack, axis=1), axis=0)

    best["all_solutions"] = solutions
    best["phase_stability"] = phase_stability
    return best


def fit_phase_chain_to_traces(
    even_weights,
    odd_weights,
    even_traces,
    odd_traces,
    even_delta_beta,
    odd_delta_beta,
    lag_step_m,
    coord_offsets,
    n_starts=8,
    random_seed=0,
    initial_amplitude_scale=None,
    initial_phase_chain=None,
):
    """
    Совместный fit нескольких соседних координат.

    Для координаты с offset = s используется окно фаз
        phase_chain[s : s + N]
    одинаковой длины N = len(weights).
    """
    even_weights = np.asarray(even_weights, dtype=np.float64).reshape(-1)
    odd_weights = np.asarray(odd_weights, dtype=np.float64).reshape(-1)
    even_traces = np.asarray(even_traces, dtype=np.float64)
    odd_traces = np.asarray(odd_traces, dtype=np.float64)
    even_delta_beta = np.asarray(even_delta_beta, dtype=np.float64).reshape(-1)
    odd_delta_beta = np.asarray(odd_delta_beta, dtype=np.float64).reshape(-1)
    coord_offsets = np.asarray(coord_offsets, dtype=np.int64).reshape(-1)

    if even_weights.size != odd_weights.size:
        raise ValueError("even_weights and odd_weights must have the same size")
    if even_traces.ndim != 2 or odd_traces.ndim != 2:
        raise ValueError("even_traces and odd_traces must be 2D arrays")
    if even_traces.shape[1] != coord_offsets.size or odd_traces.shape[1] != coord_offsets.size:
        raise ValueError("coord_offsets must match the number of coordinate traces")

    n = even_weights.size
    max_offset = int(np.max(coord_offsets))
    chain_len = n + max_offset
    n_coords = coord_offsets.size
    rng = np.random.default_rng(random_seed)

    def unpack_params(params):
        amplitude_scale = np.exp(params[0])
        phase_chain = np.concatenate([[0.0], params[1:chain_len]])
        even_offsets = params[chain_len : chain_len + n_coords]
        odd_offsets = params[chain_len + n_coords : chain_len + 2 * n_coords]
        return amplitude_scale, phase_chain, even_offsets, odd_offsets

    def residual_vector(params):
        amplitude_scale, phase_chain, even_offsets, odd_offsets = unpack_params(params)
        residuals = []
        for col, offset in enumerate(coord_offsets):
            phase_window = phase_chain[offset : offset + n]
            model_even = even_offsets[col] + intensity_from_lag_harmonics(
                even_weights,
                phase_window,
                amplitude_scale,
                even_delta_beta,
                lag_step_m,
            )
            model_odd = odd_offsets[col] + intensity_from_lag_harmonics(
                odd_weights,
                phase_window,
                amplitude_scale,
                odd_delta_beta,
                lag_step_m,
            )
            residuals.append(model_even - even_traces[:, col])
            residuals.append(model_odd - odd_traces[:, col])
        return np.concatenate(residuals)

    if initial_amplitude_scale is None:
        trace_scale = max(np.std(even_traces), np.std(odd_traces), 1e-6)
        weight_scale = max(np.sum(even_weights**2), np.sum(odd_weights**2), 1e-6)
        initial_amplitude_scale = np.sqrt(trace_scale / weight_scale)

    if initial_phase_chain is not None:
        initial_phase_chain = np.asarray(initial_phase_chain, dtype=np.float64).reshape(-1)
        if initial_phase_chain.size != chain_len:
            raise ValueError("initial_phase_chain must have length N + max(coord_offsets)")

    solutions = []
    best = None
    for start_idx in range(int(n_starts)):
        x0 = np.zeros(chain_len + 2 * n_coords, dtype=np.float64)
        x0[0] = np.log(max(initial_amplitude_scale, 1e-8))
        if initial_phase_chain is not None and start_idx == 0:
            x0[1:chain_len] = initial_phase_chain[1:]
        elif start_idx > 0:
            x0[1:chain_len] = rng.uniform(-np.pi, np.pi, size=chain_len - 1)

        x0[chain_len : chain_len + n_coords] = np.mean(even_traces, axis=0)
        x0[chain_len + n_coords : chain_len + 2 * n_coords] = np.mean(odd_traces, axis=0)

        result = least_squares(
            residual_vector,
            x0,
            method="lm",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=120,
        )

        amplitude_scale, phase_chain, even_offsets, odd_offsets = unpack_params(result.x)
        modeled_even = np.zeros_like(even_traces)
        modeled_odd = np.zeros_like(odd_traces)
        phase_windows = []
        for col, offset in enumerate(coord_offsets):
            phase_window = phase_chain[offset : offset + n]
            phase_windows.append(phase_window)
            modeled_even[:, col] = even_offsets[col] + intensity_from_lag_harmonics(
                even_weights,
                phase_window,
                amplitude_scale,
                even_delta_beta,
                lag_step_m,
            )
            modeled_odd[:, col] = odd_offsets[col] + intensity_from_lag_harmonics(
                odd_weights,
                phase_window,
                amplitude_scale,
                odd_delta_beta,
                lag_step_m,
            )

        solution = {
            "cost": float(result.cost),
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.nfev),
            "amplitude_scale": float(amplitude_scale),
            "phase_chain": phase_chain,
            "phase_windows": phase_windows,
            "coord_offsets": coord_offsets.copy(),
            "even_offsets": even_offsets,
            "odd_offsets": odd_offsets,
            "modeled_even_traces": modeled_even,
            "modeled_odd_traces": modeled_odd,
        }
        solutions.append(solution)
        if best is None or solution["cost"] < best["cost"]:
            best = solution

    solutions.sort(key=lambda item: item["cost"])
    chain_stack = np.stack(
        [np.unwrap(sol["phase_chain"]) for sol in solutions[: min(8, len(solutions))]],
        axis=0,
    )
    chain_phase_stability = np.std(chain_stack, axis=0)

    best["all_solutions"] = solutions
    best["phase_stability"] = chain_phase_stability
    return best

def colormap_raw_reflectograms(
    result,
    lambda_um,
    i_start=0,
    i_end=1000,
    t_min=0e-6,
    t_max=60e-6,
    dual_pulse=False,
    even_only=True,
):
    """
    Визуализирует тепловую карту набора рефлектограмм (без деконволюции).

    Параметры:
    - result: словарь от read_reflectograms
    - lambda_um: длина волны (мкм) для get_optical_params
    - i_start, i_end: диапазон рефлектограмм (как в исходном массиве result['data'])
    - t_min, t_max: временное окно (в секундах)
    - dual_pulse: если True, применяет фильтрацию по чётности (обычно нужно для dual-pulse режима)
    - even_only: если True, оставляет чётные; если False — нечётные (полезно для отладки)
    """

    data = result["data"]
    fs = result["sampling_rate"]

    # 1) Сначала режем по исходным индексам
    i_start = max(0, int(i_start))
    i_end = min(int(i_end), data.shape[0])
    if i_end <= i_start:
        raise ValueError(f"Bad range: i_start={i_start}, i_end={i_end}")

    reflectograms = data[i_start:i_end, :]   # shape: (K, N)
    N = reflectograms.shape[1]
    t = np.arange(N) / fs

    # 2) Окно по времени
    mask = (t >= t_min) & (t <= t_max)
    if not np.any(mask):
        raise ValueError("Time window is empty: check t_min/t_max and sampling_rate.")

    # в метры (как у тебя было)
    v_g = get_optical_params(lambda_um)[2]
    x = t[mask] * v_g / 2
    data_cut = reflectograms[:, mask]

    # 3) Dual-pulse логика: оставить только чётные (или нечётные) по ГЛОБАЛЬНОМУ индексу
    if dual_pulse:
        global_idx = np.arange(i_start, i_end)  # исходные номера рефлектограмм
        parity = 0 if even_only else 1
        sel = (global_idx % 2) == parity

        data_cut = data_cut[sel, :]
        global_idx = global_idx[sel]

        if data_cut.shape[0] == 0:
            raise ValueError("No reflectograms after parity filtering (check i_start/i_end).")

        # индекс пары (0,1,2,...) — “уже индекс пары”
        y = global_idx // 2
        y0, y1 = int(y[0]), int(y[-1])
        y_label = "Индекс пары"
        title_extra = "dual_pulse: " + ("even" if even_only else "odd")
    else:
        y0, y1 = i_start, i_end - 1
        y_label = "Номер рефлектограммы"
        title_extra = "single_pulse"

    # 4) Рисуем
    plt.figure(figsize=(12, 6))
    # extent: [x_min, x_max, y_max, y_min] чтобы сверху было "раньше/меньше"
    plt.imshow(
        data_cut,
        aspect="auto",
        cmap="inferno",
        extent=[x[0], x[-1], y1, y0],
        origin="upper",
    )
    plt.gca().invert_yaxis()
    plt.colorbar(label="Интенсивность")
    plt.xlabel("Расстояние (м)")
    plt.ylabel(y_label)
    plt.title(f"Сырые рефлектограммы ({i_start}–{i_end}) | {title_extra}")
    plt.tight_layout()
    #plt.show()



def sellmeier_n(lambda_um):
    """Вычисляет показатель преломления SiO2 по формуле Селмейера."""
    B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
    C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2

    lam2 = lambda_um**2
    n_squared = 1 + (B1 * lam2) / (lam2 - C1) + \
                   (B2 * lam2) / (lam2 - C2) + \
                   (B3 * lam2) / (lam2 - C3)
    return np.sqrt(n_squared)

def get_optical_params(lambda_um):
    """
    Возвращает: n, k [1/м], v_g [м/с]
    """
    n = sellmeier_n(lambda_um)
    lambda_m = lambda_um * 1e-6
    k = 2 * np.pi * n / lambda_m

    # численная производная
    dl = 1e-4  # 0.1 нм
    n_plus = sellmeier_n(lambda_um + dl)
    n_minus = sellmeier_n(lambda_um - dl)
    dn_dlambda = (n_plus - n_minus) / (2 * dl)

    # В формуле группового показателя lambda и d n / d lambda
    # должны быть выражены в согласованных единицах.
    c0 = 299792458.0
    vg = c0 / (n - lambda_um * dn_dlambda)

    return n, k, vg

def plot_signal_and_fft_at_z(result,
                             z_target,
                             i_start,
                             i_end,
                             lambda_um=1.55,
                             dz_avg=0.0,
                             remove_dc=False,
                             window='hann',
                             show=True):
    """
    Достаёт сигнал в координате z_target по трассам [i_start:i_end)
    и строит его амплитудный спектр по номеру рефлектограммы.

    Параметры
    ---------
    result : dict
        Выход read_reflectograms(...).
    z_target : float (м)
        Координата вдоль волокна.
    i_start, i_end : int
        Диапазон номеров рефлектограмм (i_end не включительно).
    lambda_um : float
        Длина волны лазера (в мкм) для расчёта групповой скорости.
    dz_avg : float
        Полуширина окна усреднения по координате (м). 0 — без усреднения.
    remove_dc : bool
        Вычитать ли среднее (DC) из временного ряда перед Фурье.
    window : {'hann', None}
        Окно для спектрального анализа.
    show : bool
        Рисовать ли графики.

    Возвращает
    ----------
    traces_idx : np.ndarray
        Индексы трасс (ось времени).
    y : np.ndarray
        Сигнал в выбранной координате по трассам.
    f_hz : np.ndarray
        Ось частот (Гц).
    Y : np.ndarray (комплекс)
        Спектр rFFT(y_win).
    """

    data = result['data'][i_start:i_end, :]       # (T, N_time)
    fs = result['sampling_rate']                  # частота дискретизации внутри трассы
    scan_rate = result['scan_rate']               # Гц — частота снятия рефлектограмм (ось времени)
    vg = get_optical_params(lambda_um)[2]         # групповая скорость

    # координатная ось для одной трассы
    n = np.arange(data.shape[1])
    t_samp = n / fs
    z_axis = t_samp * vg / 2

    # найдём индекс(ы) ближайших точек к z_target
    if dz_avg and dz_avg > 0:
        mask = (z_axis >= z_target - dz_avg) & (z_axis <= z_target + dz_avg)
        if not np.any(mask):
            raise ValueError("Окно dz_avg не попало в диапазон данных.")
        y = data[:, mask].mean(axis=1)
    else:
        z_idx = np.argmin(np.abs(z_axis - z_target))
        y = data[:, z_idx]

    # предобработка
    if remove_dc:
        y = y - np.mean(y)

    if window == 'hann':
        w = np.hanning(len(y))
        y_win = y * w
    else:
        y_win = y

    # спектр по оси «номер трассы»
    Y = np.fft.rfft(y_win)
    f_hz = np.fft.rfftfreq(len(y_win), d=1.0/scan_rate)

    traces_idx = np.arange(i_start, i_end)

    if show:
        fig, axs = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

        axs[0].plot(traces_idx, y, lw=1)
        axs[0].set_xlabel("Номер рефлектограммы")
        axs[0].set_ylabel("Сигнал (отн. ед.)")
        axs[0].set_title(f"z ≈ {z_target:.3f} м"
                         + (f", уср. ±{dz_avg:.3f} м" if dz_avg and dz_avg > 0 else ""))

        axs[1].plot(f_hz, np.abs(Y))
        axs[1].set_xlim(0, f_hz.max())
        axs[1].set_xlabel("Частота (Гц)")
        axs[1].set_ylabel("|FFT|")
        axs[1].set_title("Амплитудный спектр по номеру трассы")
        axs[1].grid(True)

        plt.show()

    return traces_idx, y, f_hz, Y



def estimate_laser_shift_coeff_second_difference(
    arr,
    template,
    fit_mask=None,
    weights=None,
    remove_mean_from_template=True,
    smooth_d2a=0,
):
    """
    Оценивает a[i] из второй разности по времени без построения огромных матриц.

    Модель:
        arr_corr[i, j] = arr[i, j] - a[i] * template[j]

    Идея:
        1) считаем вторую разность по времени от arr
        2) проецируем её на template -> получаем d2a_hat
        3) дважды интегрируем d2a_hat, получая a up to const + linear trend
        4) убираем среднее и линейный тренд из a

    Параметры
    ---------
    arr : ndarray, shape (N, M)
    template : ndarray, shape (M,)
    fit_mask : ndarray bool, optional
    weights : ndarray, optional
    remove_mean_from_template : bool
    smooth_d2a : int
        Если > 1, сглаживает d2a_hat простым окном.

    Возвращает
    ----------
    a : ndarray, shape (N,)
    arr_corr : ndarray, shape (N, M)
    d2a_hat : ndarray, shape (N-2,)
    """
    arr = np.asarray(arr)
    template = np.asarray(template, dtype=np.float64)

    N, M = arr.shape
    if template.shape[0] != M:
        raise ValueError("template must have shape (M,)")

    if fit_mask is None:
        fit_mask = np.ones(M, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)

    s = template[fit_mask].copy()
    X = arr[:, fit_mask]

    if remove_mean_from_template:
        s = s - np.mean(s)

    if weights is None:
        w = np.ones_like(s)
    else:
        w = np.asarray(weights, dtype=np.float64)[fit_mask]

    sw = s * np.sqrt(w)

    denom = np.dot(sw, sw)
    if denom <= 0:
        raise ValueError("Degenerate template on fit_mask")

    # Эквивалентная, но значительно более экономная по памяти форма:
    # сначала проектируем каждую рефлектограмму на шаблон,
    # затем берём вторую разность уже у одномерного ряда.
    weighted_template = s * w
    proj = X @ weighted_template / denom
    d2a_hat = proj[2:] - 2.0 * proj[1:-1] + proj[:-2]

    # Небольшое сглаживание при желании
    if smooth_d2a and smooth_d2a > 1:
        kernel = np.ones(smooth_d2a, dtype=np.float64) / smooth_d2a
        d2a_hat = np.convolve(d2a_hat, kernel, mode="same")

    # Восстановление a из второй разности:
    # a[i+1] - 2a[i] + a[i-1] = d2a_hat[i-1]
    #
    # Берём a[0]=0, a[1]=0, дальше рекурсия:
    a = np.zeros(N, dtype=np.float64)
    for k in range(N - 2):
        a[k + 2] = d2a_hat[k] + 2.0 * a[k + 1] - a[k]

    # Убираем произвольные const + linear trend
    i = np.arange(N, dtype=np.float64)
    p = np.polyfit(i, a, deg=1)
    a = a - np.polyval(p, i)

    # Дополнительно центрируем
    a = a - np.mean(a)

    arr_corr = np.array(arr, dtype=np.float32, copy=True)
    arr_corr -= (
        np.asarray(a, dtype=np.float32)[:, None]
        * np.asarray(template, dtype=np.float32)[None, :]
    )
    return a, arr_corr, d2a_hat


def estimate_laser_shift_coeff_direct(
    arr,
    template,
    fit_mask=None,
    weights=None,
    remove_mean_from_template=True,
):
    """
    Прямая покадровая МНК-оценка коэффициента a[i] в модели
        arr[i, j] ≈ background[i, j] + a[i] * template[j]

    Возвращает коэффициент и массив после вычитания шаблона.
    """
    arr = np.asarray(arr)
    template = np.asarray(template, dtype=np.float64)

    N, M = arr.shape
    if template.shape[0] != M:
        raise ValueError("template must have shape (M,)")

    if fit_mask is None:
        fit_mask = np.ones(M, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)

    s = template[fit_mask].copy()
    X = arr[:, fit_mask]

    if remove_mean_from_template:
        s = s - np.mean(s)

    if weights is None:
        w = np.ones_like(s)
    else:
        w = np.asarray(weights, dtype=np.float64)[fit_mask]

    weighted_template = s * w
    denom = np.dot(s, weighted_template)
    if denom <= 0:
        raise ValueError("Degenerate template on fit_mask")

    a = (X @ weighted_template) / denom

    arr_corr = np.array(arr, dtype=np.float32, copy=True)
    arr_corr -= (
        np.asarray(a, dtype=np.float32)[:, None]
        * np.asarray(template, dtype=np.float32)[None, :]
    )
    return a, arr_corr


def estimate_laser_shift_coeff_with_nuisance_modes(
    arr,
    template,
    fit_mask=None,
    weights=None,
    remove_mean_from_template=True,
    n_nuisance=3,
    max_mode_traces=20000,
):
    """
    Совместная МНК-оценка:
        arr[i, j] ≈ a[i] * template[j] + sum_k b_k[i] * u_k[j] + residual

    Здесь u_k[j] — дополнительные пространственные моды, извлечённые из данных
    и ортогонализованные к template. Это позволяет отделять лазерный вклад
    от других изменений формы рефлектограммы.

    Возвращает
    ----------
    a : ndarray, shape (N,)
        Коэффициент именно лазерного шаблона.
    arr_corr : ndarray, shape (N, M)
        Массив после вычитания только лазерного вклада.
    nuisance_modes_full : ndarray, shape (K, M)
        Дополнительные пространственные моды на полной оси.
    nuisance_coeffs : ndarray, shape (N, K)
        Их временные коэффициенты.
    """
    arr = np.asarray(arr)
    template = np.asarray(template, dtype=np.float64)

    N, M = arr.shape
    if template.shape[0] != M:
        raise ValueError("template must have shape (M,)")

    if fit_mask is None:
        fit_mask = np.ones(M, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)

    X = np.asarray(arr[:, fit_mask], dtype=np.float64)
    s = template[fit_mask].astype(np.float64, copy=True)

    if remove_mean_from_template:
        s = s - np.mean(s)

    if weights is None:
        w = np.ones_like(s)
    else:
        w = np.asarray(weights, dtype=np.float64)[fit_mask]

    denom_s = np.dot(s * w, s)
    if denom_s <= 0:
        raise ValueError("Degenerate template on fit_mask")

    # Первичная оценка лазерного вклада, чтобы выделить прочие изменения формы.
    a0 = (X @ (s * w)) / denom_s
    R = X - a0[:, None] * s[None, :]

    # Убираем среднее по времени для извлечения именно изменяющихся мод.
    R = R - np.mean(R, axis=0, keepdims=True)

    nuisance_modes = np.zeros((0, s.size), dtype=np.float64)
    nuisance_coeffs = np.zeros((N, 0), dtype=np.float64)

    if n_nuisance and n_nuisance > 0:
        if max_mode_traces and R.shape[0] > max_mode_traces:
            stride = int(np.ceil(R.shape[0] / max_mode_traces))
            R_modes = R[::stride]
        else:
            R_modes = R

        # Для длинной серии трасс выгоднее искать моды через пространственную
        # ковариационную матрицу размера (M_fit x M_fit), а не через полный SVD
        # массива (N x M_fit).
        cov = (R_modes.T @ R_modes) / max(1, R_modes.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        candidates = eigvecs[:, order[: min(n_nuisance, eigvecs.shape[1])]].T.copy()

        # Грамм-Шмидт с весами: моды не должны содержать компоненту лазерного шаблона
        # и должны быть взаимно ортогональны.
        orth_modes = []
        s_norm = np.sqrt(denom_s)
        s_unit = s / s_norm

        for mode in candidates:
            mode = mode - np.dot(mode * w, s_unit) * s_unit
            for prev in orth_modes:
                mode = mode - np.dot(mode * w, prev) * prev

            mode_norm = np.sqrt(np.dot(mode * w, mode))
            if mode_norm > 1e-12:
                orth_modes.append(mode / mode_norm)

        if orth_modes:
            nuisance_modes = np.vstack(orth_modes)

    # Совместный базис: первая колонка — лазерный шаблон.
    G_cols = [s]
    if nuisance_modes.shape[0] > 0:
        G_cols.extend(list(nuisance_modes))
    G = np.column_stack(G_cols)  # shape (M_fit, 1 + K)

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[None, :]
    Gw = G * sqrt_w[:, None]

    gram = Gw.T @ Gw
    gram_inv = np.linalg.pinv(gram, rcond=1e-12)
    coeffs = Xw @ Gw @ gram_inv

    a = coeffs[:, 0]
    nuisance_coeffs = coeffs[:, 1:] if coeffs.shape[1] > 1 else np.zeros((N, 0), dtype=np.float64)

    arr_corr = np.array(arr, dtype=np.float32, copy=True)
    arr_corr -= (
        np.asarray(a, dtype=np.float32)[:, None]
        * np.asarray(template, dtype=np.float32)[None, :]
    )

    nuisance_modes_full = np.zeros((nuisance_modes.shape[0], M), dtype=np.float64)
    if nuisance_modes.shape[0] > 0:
        nuisance_modes_full[:, fit_mask] = nuisance_modes

    return a, arr_corr, nuisance_modes_full, nuisance_coeffs

from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


def build_laser_sensitivity_template_from_osc_csv(
    osc_csv_path,
    z_main,
    normalize=True,
    smooth_points=9,
):
    """
    Строит пространственный шаблон s(z) температурного сдвига лазера
    по oscillation csv через первую главную компоненту.

    Параметры
    ---------
    osc_csv_path : str
        Путь к csv вида time_s, amp_340.000, amp_340.501, ...
    z_main : np.ndarray
        Координатная ось основной рефлектограммы (в метрах),
        на которую надо интерполировать шаблон.
    normalize : bool
        Нормировать ли шаблон на RMS=1.
    smooth_points : int
        Размер медианного сглаживания по z для подавления шума в шаблоне.

    Возвращает
    ----------
    s_main : np.ndarray shape (len(z_main),)
        Интерполированный шаблон чувствительности.
    z_osc : np.ndarray
        Координаты из osc-файла.
    s_osc : np.ndarray
        Шаблон на родной сетке osc-файла.
    df : pd.DataFrame
        Загруженный csv.
    """
    df = pd.read_csv(osc_csv_path)

    amp_cols = [c for c in df.columns if c.startswith("amp_")]
    if len(amp_cols) == 0:
        raise ValueError("В csv не найдены колонки amp_*")

    # Координаты вида amp_340.000 -> 340.000
    z_osc = np.array([float(c.split("amp_")[1]) for c in amp_cols], dtype=float)

    # Матрица: время x координата
    X = df[amp_cols].to_numpy(dtype=float)

    # Убираем среднее по времени в каждой координате
    Xc = X - np.mean(X, axis=0, keepdims=True)

    # Первая главная компонента по пространству
    # Xc = U S Vt, первая строка Vt[0] = главный пространственный паттерн
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    s_osc = Vt[0].copy()

    # Знак PCA произволен; можно зафиксировать, чтобы было "положительное в среднем"
    if np.nanmean(s_osc) < 0:
        s_osc *= -1.0

    # Небольшое подавление пилообразного шума
    if smooth_points and smooth_points > 1:
        s_osc = median_filter(s_osc, size=smooth_points, mode="nearest")

    # Нормировка
    if normalize:
        rms = np.sqrt(np.mean(s_osc**2))
        if rms > 0:
            s_osc = s_osc / rms

    # Интерполяция на ось основной рефы
    f = interp1d(
        z_osc,
        s_osc,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    s_main = f(z_main)

    # Повторная нормировка после интерполяции
    if normalize:
        rms = np.sqrt(np.mean(s_main**2))
        if rms > 0:
            s_main = s_main / rms

    return s_main, z_osc, s_osc, df


def build_laser_mode_basis_from_osc_csv(
    osc_csv_path,
    z_main,
    n_modes=3,
    normalize=True,
    smooth_points=9,
):
    """
    Строит несколько пространственных лазерных мод из osc csv.

    В отличие от build_laser_sensitivity_template_from_osc_csv,
    здесь используется не только первая PCA-компонента, а целый базис.
    """
    df = pd.read_csv(osc_csv_path)

    amp_cols = [c for c in df.columns if c.startswith("amp_")]
    if len(amp_cols) == 0:
        raise ValueError("В csv не найдены колонки amp_*")

    z_osc = np.array([float(c.split("amp_")[1]) for c in amp_cols], dtype=float)
    X = df[amp_cols].to_numpy(dtype=float)
    Xc = X - np.mean(X, axis=0, keepdims=True)

    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    n_keep = min(int(n_modes), vt.shape[0])
    basis_osc = vt[:n_keep].T.copy()

    if smooth_points and smooth_points > 1:
        for idx in range(basis_osc.shape[1]):
            basis_osc[:, idx] = median_filter(basis_osc[:, idx], size=smooth_points, mode="nearest")

    basis_main = np.zeros((len(z_main), n_keep), dtype=float)
    for idx in range(n_keep):
        mode = basis_osc[:, idx].copy()
        if np.nanmean(mode) < 0:
            mode *= -1.0

        interp = interp1d(
            z_osc,
            mode,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        basis_main[:, idx] = interp(z_main)

    # Ортогонализуем моды уже на рабочей сетке.
    orth_modes = []
    for idx in range(basis_main.shape[1]):
        mode = basis_main[:, idx].copy()
        for prev in orth_modes:
            mode = mode - np.dot(mode, prev) * prev

        norm = np.linalg.norm(mode)
        if norm > 1e-12:
            orth_modes.append(mode / norm)

    if not orth_modes:
        raise ValueError("Не удалось построить ненулевые лазерные моды")

    basis_main = np.column_stack(orth_modes)

    if normalize:
        for idx in range(basis_main.shape[1]):
            rms = np.sqrt(np.mean(basis_main[:, idx] ** 2))
            if rms > 0:
                basis_main[:, idx] /= rms

    return basis_main, z_osc, basis_osc, df


def subtract_mode_basis(
    arr,
    basis,
    fit_mask=None,
    weights=None,
):
    """
    Для каждой рефлектограммы оценивает коэффициенты мод в basis
    и вычитает восстановленный вклад этих мод.
    """
    arr = np.asarray(arr)
    basis = np.asarray(basis, dtype=np.float64)

    N, M = arr.shape
    if basis.ndim != 2 or basis.shape[0] != M:
        raise ValueError("basis must have shape (M, K)")

    if fit_mask is None:
        fit_mask = np.ones(M, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)

    X = np.asarray(arr[:, fit_mask], dtype=np.float64)
    B = basis[fit_mask, :].copy()

    if weights is None:
        w = np.ones(B.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)[fit_mask]

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[None, :]
    Bw = B * sqrt_w[:, None]

    gram = Bw.T @ Bw
    gram_inv = np.linalg.pinv(gram, rcond=1e-12)
    coeffs = Xw @ Bw @ gram_inv

    model = coeffs @ basis.T
    arr_corr = np.array(arr, dtype=np.float32, copy=True)
    arr_corr -= np.asarray(model, dtype=np.float32)

    return coeffs, arr_corr


def build_time_dependent_template_from_osc_csv(
    osc_csv_path,
    z_main,
    time_main,
    smooth_points=9,
    zero_time_at_start=True,
    normalize_each_time=True,
):
    """
    Интерполирует карту oscillation csv на оси (time_main, z_main).

    Возвращает
    ----------
    templates_main : ndarray, shape (len(time_main), len(z_main))
        Времязависимый шаблон лазерной чувствительности.
    z_osc : ndarray
    t_osc : ndarray
    X_osc : ndarray
        Исходная матрица csv на родной сетке.
    df : pd.DataFrame
    """
    df = pd.read_csv(osc_csv_path)

    amp_cols = [c for c in df.columns if c.startswith("amp_")]
    if len(amp_cols) == 0:
        raise ValueError("В csv не найдены колонки amp_*")

    z_osc = np.array([float(c.split("amp_")[1]) for c in amp_cols], dtype=float)
    t_osc = df["time_s"].to_numpy(dtype=float)
    if zero_time_at_start and t_osc.size > 0:
        t_osc = t_osc - t_osc[0]
    X_osc = df[amp_cols].to_numpy(dtype=float)

    if smooth_points and smooth_points > 1:
        for idx in range(X_osc.shape[0]):
            X_osc[idx] = median_filter(X_osc[idx], size=smooth_points, mode="nearest")

    if normalize_each_time:
        X_osc = X_osc - np.mean(X_osc, axis=1, keepdims=True)
        rms = np.sqrt(np.mean(X_osc ** 2, axis=1, keepdims=True))
        rms[rms <= 1e-12] = 1.0
        X_osc = X_osc / rms

    # Сначала интерполяция по z для каждого csv-времени.
    X_z = np.zeros((X_osc.shape[0], len(z_main)), dtype=float)
    for idx in range(X_osc.shape[0]):
        interp_z = interp1d(
            z_osc,
            X_osc[idx],
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        X_z[idx] = interp_z(z_main)

    # Затем по времени для каждого z.
    templates_main = np.zeros((len(time_main), len(z_main)), dtype=float)
    for j in range(len(z_main)):
        interp_t = interp1d(
            t_osc,
            X_z[:, j],
            kind="linear",
            bounds_error=False,
            fill_value=(X_z[0, j], X_z[-1, j]),
        )
        templates_main[:, j] = interp_t(time_main)

    if normalize_each_time:
        templates_main = templates_main - np.mean(templates_main, axis=1, keepdims=True)
        rms = np.sqrt(np.mean(templates_main ** 2, axis=1, keepdims=True))
        rms[rms <= 1e-12] = 1.0
        templates_main = templates_main / rms

    return templates_main, z_osc, t_osc, X_osc, df


def subtract_time_dependent_template(
    arr,
    templates,
    fit_mask=None,
    weights=None,
    baseline_window_traces=20000,
    center_templates=True,
    baseline=None,
):
    """
    Модель:
        arr(t, z) ≈ baseline_slow(t, z) + a(t) * template(t, z) + residual

    baseline_slow используется только для устойчивой оценки a(t),
    а из финального массива вычитается только лазерный вклад a(t)*template(t, z).
    """
    arr = np.asarray(arr, dtype=np.float32)
    templates = np.asarray(templates, dtype=np.float64)

    if templates.shape != arr.shape:
        raise ValueError("templates must have the same shape as arr")

    N, M = arr.shape
    if fit_mask is None:
        fit_mask = np.ones(M, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)

    if weights is None:
        w = np.ones(np.count_nonzero(fit_mask), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)[fit_mask]

    if baseline is None:
        if baseline_window_traces and baseline_window_traces > 1:
            baseline = uniform_filter1d(arr, size=int(baseline_window_traces), axis=0, mode="nearest")
        else:
            baseline = np.zeros_like(arr)
    else:
        baseline = np.asarray(baseline, dtype=np.float32)
        if baseline.shape != arr.shape:
            raise ValueError("baseline must have the same shape as arr")

    residual = np.asarray(arr - baseline, dtype=np.float64)
    a = np.zeros(N, dtype=np.float64)

    for idx in range(N):
        s = templates[idx, fit_mask].copy()
        if center_templates:
            s = s - np.mean(s)

        denom = np.dot(s * w, s)
        if denom > 1e-12:
            a[idx] = np.dot(residual[idx, fit_mask] * w, s) / denom

    arr_corr = np.array(arr, dtype=np.float32, copy=True)
    arr_corr -= np.asarray(a[:, None] * templates, dtype=np.float32)
    return a, arr_corr, baseline


def build_piecewise_time_baseline(
    arr,
    time_main,
    sample_times,
    half_window_s=None,
):
    """
    Оценивает медленно меняющийся baseline на заданных sample_times
    как локальные средние по данным, затем интерполирует его во времени.
    """
    arr = np.asarray(arr, dtype=np.float32)
    time_main = np.asarray(time_main, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)

    if half_window_s is None:
        if sample_times.size > 1:
            half_window_s = 0.5 * np.median(np.diff(sample_times))
        else:
            half_window_s = 0.5

    anchors = np.zeros((sample_times.size, arr.shape[1]), dtype=np.float32)
    for idx, t0 in enumerate(sample_times):
        mask = (time_main >= t0 - half_window_s) & (time_main <= t0 + half_window_s)
        if not np.any(mask):
            nearest = np.argmin(np.abs(time_main - t0))
            anchors[idx] = arr[nearest]
        else:
            anchors[idx] = np.mean(arr[mask], axis=0)

    baseline = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        interp_t = interp1d(
            sample_times,
            anchors[:, j],
            kind="linear",
            bounds_error=False,
            fill_value=(anchors[0, j], anchors[-1, j]),
        )
        baseline[:, j] = interp_t(time_main)

    return baseline, anchors


def estimate_modulation_frequency_from_residual(
    residual,
    scan_rate,
    fit_mask=None,
    freq_min=0.5,
    freq_max=2.0,
    decimate=10,
    coord_stride=1,
):
    """
    Оценивает доминирующую частоту модуляции по baseline-removed данным.

    Усредняет спектральную мощность по координатам, чтобы не зависеть
    от локального знака чувствительности.
    """
    residual = np.asarray(residual, dtype=np.float32)
    N, M = residual.shape

    if fit_mask is None:
        fit_mask = np.ones(M, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)

    decimate = max(1, int(decimate))
    coord_stride = max(1, int(coord_stride))

    data = residual[::decimate, fit_mask]
    eff_scan_rate = scan_rate / decimate

    freq_axis = np.fft.rfftfreq(data.shape[0], d=1.0 / eff_scan_rate)
    power = np.zeros(freq_axis.shape[0], dtype=np.float64)

    for col in range(0, data.shape[1], coord_stride):
        x = np.asarray(data[:, col], dtype=np.float64)
        x = x - np.mean(x)
        X = np.fft.rfft(x)
        power += np.abs(X) ** 2

    band = (freq_axis >= freq_min) & (freq_axis <= freq_max)
    if not np.any(band):
        raise ValueError("Empty frequency search band")

    idx_local = np.argmax(power[band])
    f0 = float(freq_axis[band][idx_local])
    return f0, freq_axis, power


def reconstruct_lockin_component_from_residual(
    residual,
    scan_rate,
    modulation_freq_hz,
    envelope_window_periods=2.0,
):
    """
    Восстанавливает узкополосную модуляцию около modulation_freq_hz
    методом комплексного детектирования (lock-in).

    Предполагается, что амплитуда/чувствительность по координате
    меняются медленно относительно периода модуляции.
    """
    residual = np.asarray(residual, dtype=np.float32)
    N, M = residual.shape

    if modulation_freq_hz <= 0:
        raise ValueError("modulation_freq_hz must be positive")

    window_traces = max(
        3,
        int(round(envelope_window_periods * scan_rate / modulation_freq_hz)),
    )

    time_axis = np.arange(N, dtype=np.float64) / scan_rate
    carrier_cos = np.cos(2.0 * np.pi * modulation_freq_hz * time_axis)
    carrier_sin = np.sin(2.0 * np.pi * modulation_freq_hz * time_axis)

    component = np.zeros_like(residual, dtype=np.float32)
    amplitude = np.zeros_like(residual, dtype=np.float32)

    for j in range(M):
        x = np.asarray(residual[:, j], dtype=np.float64)

        i_mix = uniform_filter1d(x * carrier_cos, size=window_traces, mode="nearest")
        q_mix = uniform_filter1d(x * carrier_sin, size=window_traces, mode="nearest")

        # Умножение на 2 компенсирует смешение cos/sin компоненты.
        i_env = 2.0 * i_mix
        q_env = 2.0 * q_mix

        component[:, j] = (
            i_env * carrier_cos + q_env * carrier_sin
        ).astype(np.float32)
        amplitude[:, j] = np.sqrt(i_env ** 2 + q_env ** 2).astype(np.float32)

    return component, amplitude, window_traces


def plot_reflectogram_heatmap(
    arr,
    z_axis,
    i_start=0,
    i_end=None,
    z_min=None,
    z_max=None,
    title="Тепловая карта рефлектограмм",
    cmap="inferno",
    vmin=None,
    vmax=None,
    subtract_row_mean=False,
    show=True,
    y_axis=None,
    y_label="Номер рефлектограммы",
):
    """
    Рисует тепловую карту массива arr shape = (n_refl, n_z)

    Параметры
    ---------
    arr : np.ndarray
        Матрица рефлектограмм (номер рефы, координата)
    z_axis : np.ndarray
        Координата в метрах, shape = (n_z,)
    i_start, i_end : int
        Диапазон рефлектограмм
    z_min, z_max : float
        Окно по координате
    subtract_row_mean : bool
        Если True, вычитает среднее из каждой рефы отдельно.
        Полезно, если хочется лучше видеть мелкие изменения.
    """

    arr = np.asarray(arr)
    z_axis = np.asarray(z_axis)

    if i_end is None:
        i_end = arr.shape[0]

    i_start = max(0, int(i_start))
    i_end = min(int(i_end), arr.shape[0])

    # выбор диапазона по z
    if z_min is None:
        z_min = z_axis[0]
    if z_max is None:
        z_max = z_axis[-1]

    z_mask = (z_axis >= z_min) & (z_axis <= z_max)
    if not np.any(z_mask):
        raise ValueError("Пустой диапазон по z")

    data = arr[i_start:i_end, :][:, z_mask].copy()
    z_cut = z_axis[z_mask]

    if subtract_row_mean:
        data -= np.mean(data, axis=1, keepdims=True)

    if y_axis is None:
        y_axis = np.arange(arr.shape[0], dtype=float)
    else:
        y_axis = np.asarray(y_axis, dtype=float)
        if y_axis.shape[0] != arr.shape[0]:
            raise ValueError("y_axis must have the same length as arr.shape[0]")

    y_cut = y_axis[i_start:i_end]

    plt.figure(figsize=(12, 6))
    plt.imshow(
        data,
        aspect="auto",
        origin="upper",
        extent=[z_cut[0], z_cut[-1], y_cut[-1], y_cut[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.gca().invert_yaxis()
    plt.colorbar(label="Амплитуда")
    plt.xlabel("Координата z, м")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    if show:
        plt.show()
