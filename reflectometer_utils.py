import numpy as np

from raw_data import sellmeier_n


LIGHT_SPEED_M_PER_S = 299792458.0


def distance_axis_from_sampling_rate(n_samples, sampling_rate_hz, wavelength_um=1.55):
    n_eff = float(sellmeier_n(wavelength_um))
    distance_step_m = LIGHT_SPEED_M_PER_S / (2.0 * n_eff * float(sampling_rate_hz))
    return np.arange(n_samples, dtype=np.float64) * distance_step_m, distance_step_m, n_eff


def baseline_mask_from_tail(distance_axis_m, tail_length_m):
    distance_axis_m = np.asarray(distance_axis_m, dtype=np.float64).reshape(-1)
    if distance_axis_m.size == 0:
        raise ValueError("distance_axis_m is empty")
    if tail_length_m is None or float(tail_length_m) <= 0.0:
        raise ValueError("tail_length_m must be positive")
    z_max = float(distance_axis_m[-1])
    z_min = z_max - float(tail_length_m)
    mask = distance_axis_m >= z_min
    if not np.any(mask):
        raise ValueError("Baseline tail window is empty")
    return mask, float(distance_axis_m[mask][0]), float(distance_axis_m[mask][-1])


def subtract_trace_baseline_from_tail(trace_matrix, distance_axis_m, tail_length_m):
    data = np.asarray(trace_matrix, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("trace_matrix must be 2D: traces x samples")
    mask, z_min, z_max = baseline_mask_from_tail(distance_axis_m, tail_length_m)
    baseline_per_trace = np.mean(data[:, mask], axis=1, keepdims=True)
    corrected = data - baseline_per_trace
    return corrected, baseline_per_trace[:, 0], mask, z_min, z_max
