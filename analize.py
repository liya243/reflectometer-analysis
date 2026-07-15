import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

from raw_data import (
    extract_even_odd_pulse_weights,
    fit_phase_chain_to_traces,
    fit_real_harmonics_vs_delta_beta,
    fit_shared_phases_from_harmonics,
    fit_shared_phases_to_traces,
    interpolate_sweep_read_value_per_reflectogram,
    read_reflectograms,
    real_fit_to_complex_harmonics,
    sellmeier_n,
)


def phase_diff_stats_from_single_solutions(solutions):
    top = solutions[: min(8, len(solutions))]
    phase_matrix = np.stack([np.unwrap(sol["phases"]) for sol in top], axis=0)
    phase_diff_matrix = np.diff(phase_matrix, axis=1)
    return np.mean(phase_diff_matrix, axis=0), np.std(phase_diff_matrix, axis=0)


def phase_diff_stats_from_joint_solutions(solutions, local_idx):
    top = solutions[: min(8, len(solutions))]
    phase_matrix = np.stack(
        [np.unwrap(sol["phase_windows"][local_idx]) for sol in top],
        axis=0,
    )
    phase_diff_matrix = np.diff(phase_matrix, axis=1)
    return np.mean(phase_diff_matrix, axis=0), np.std(phase_diff_matrix, axis=0)


def build_initial_phase_chain(single_phase_seed, central_offset, chain_len):
    initial_phase_chain = np.full(chain_len, np.nan, dtype=np.float64)
    initial_phase_chain[central_offset : central_offset + single_phase_seed.size] = single_phase_seed

    left_step = single_phase_seed[1] - single_phase_seed[0]
    for idx in range(central_offset - 1, -1, -1):
        initial_phase_chain[idx] = initial_phase_chain[idx + 1] - left_step

    right_step = single_phase_seed[-1] - single_phase_seed[-2]
    for idx in range(central_offset + single_phase_seed.size, chain_len):
        initial_phase_chain[idx] = initial_phase_chain[idx - 1] + right_step

    initial_phase_chain -= initial_phase_chain[0]
    return initial_phase_chain


path_dat = "0 - 350 m calibration.dat"
path_sweep = "sweep.csv"
scan_rate_hz = 10000

lambda0_nm = 1550.0
lambda0_m = lambda0_nm * 1e-9
n_eff = float(sellmeier_n(1.55))

z_min = 0.0
z_max = 350.0
pulse_z_min = 50.0
pulse_z_max = 75.0
pulse_baseline_z = 40.0
useful_z_min = 100.0
useful_z_max = 260.0
pm_per_unit = 0.018

pulse_threshold_fraction = 0.01
harmonic_starts = 6
trace_starts = 8
harmonic_stride = 32
trace_fit_stride = 32
joint_half_spans = [1, 2, 3]

save_dir = Path("analysis_outputs")
save_dir.mkdir(exist_ok=True)
for old_file in save_dir.iterdir():
    if old_file.is_file():
        old_file.unlink()

result = read_reflectograms(path_dat, scan_rate=scan_rate_hz)
arr = result["data"]
n_traces, n_coords = arr.shape
z = np.linspace(z_min, z_max, n_coords, dtype=np.float64)

time_axis_s, sweep_units, wavelength_shift_pm, _ = interpolate_sweep_read_value_per_reflectogram(
    result,
    path_sweep,
    pm_per_unit=pm_per_unit,
    use_column="read_value",
)

delta_lambda_m = wavelength_shift_pm * 1e-12
delta_beta = -2.0 * np.pi * n_eff * delta_lambda_m / (lambda0_m**2)

pulse_info = extract_even_odd_pulse_weights(
    arr,
    z,
    pulse_z_min=pulse_z_min,
    pulse_z_max=pulse_z_max,
    baseline_z=pulse_baseline_z,
    threshold_fraction=pulse_threshold_fraction,
)
support_z = pulse_info["z"]
even_weights = pulse_info["even_weights"]
odd_weights = pulse_info["odd_weights"]
lag_step_m = float(np.median(np.diff(support_z)))

sweep_slope_units_per_s = np.polyfit(time_axis_s, sweep_units, 1)[0]
required_delta_beta = np.pi / lag_step_m
required_delta_lambda_m = required_delta_beta * lambda0_m**2 / (2.0 * np.pi * n_eff)
required_delta_lambda_pm = abs(required_delta_lambda_m) * 1e12
required_units = required_delta_lambda_pm / pm_per_unit
window_seconds = required_units / abs(sweep_slope_units_per_s)
window_reflectograms = int(round(window_seconds * scan_rate_hz))
if window_reflectograms % 2 == 1:
    window_reflectograms += 1

window_center = n_traces // 2
window_start = max(0, window_center - window_reflectograms // 2)
window_end = min(n_traces, window_start + window_reflectograms)
window_start = max(0, window_end - window_reflectograms)

useful_mask = (z >= useful_z_min) & (z <= useful_z_max)
useful_z = z[useful_mask]
useful = arr[:, useful_mask] - arr[:, [-1]]
useful_window = useful[window_start:window_end]
delta_beta_window = delta_beta[window_start:window_end]
time_window_s = time_axis_s[window_start:window_end]

even_signal = useful_window[0::2]
odd_signal = useful_window[1::2]
even_delta_beta = delta_beta_window[0::2]
odd_delta_beta = delta_beta_window[1::2]

combined_std = np.std(even_signal, axis=0) + np.std(odd_signal, axis=0)
target_idx = int(np.argmax(combined_std))
target_z = float(useful_z[target_idx])
target_even_trace = even_signal[:, target_idx]
target_odd_trace = odd_signal[:, target_idx]

target_even_trace_fit = target_even_trace[::harmonic_stride]
target_odd_trace_fit = target_odd_trace[::harmonic_stride]
even_delta_beta_fit = even_delta_beta[::harmonic_stride]
odd_delta_beta_fit = odd_delta_beta[::harmonic_stride]

print("Starting harmonic initialization", flush=True)
max_lag = min(even_weights.size - 1, 8)
lag_indices = np.arange(1, max_lag + 1, dtype=int)
L_values_m = lag_indices.astype(np.float64) * lag_step_m

even_fit = fit_real_harmonics_vs_delta_beta(
    target_even_trace_fit[:, None],
    even_delta_beta_fit,
    L_values_m,
    include_constant=True,
)
odd_fit = fit_real_harmonics_vs_delta_beta(
    target_odd_trace_fit[:, None],
    odd_delta_beta_fit,
    L_values_m,
    include_constant=True,
)
target_even_h = real_fit_to_complex_harmonics(even_fit)[:, 0]
target_odd_h = real_fit_to_complex_harmonics(odd_fit)[:, 0]

harmonic_init = fit_shared_phases_from_harmonics(
    even_weights,
    odd_weights,
    target_even_h,
    target_odd_h,
    n_starts=harmonic_starts,
    random_seed=0,
)
print("Harmonic initialization finished", flush=True)

print("Starting single-coordinate trace uniqueness fit", flush=True)
single_fit = fit_shared_phases_to_traces(
    even_weights,
    odd_weights,
    target_even_trace[::trace_fit_stride],
    target_odd_trace[::trace_fit_stride],
    even_delta_beta[::trace_fit_stride],
    odd_delta_beta[::trace_fit_stride],
    lag_step_m,
    n_starts=trace_starts,
    random_seed=0,
    initial_amplitude_scale=harmonic_init["amplitude_scale"],
    initial_phases=harmonic_init["phases"],
)
print("Single-coordinate trace fit finished", flush=True)

single_phase_diff_mean, single_phase_diff_std = phase_diff_stats_from_single_solutions(
    single_fit["all_solutions"]
)

single_phase_seed = np.unwrap(single_fit["phases"])
single_model_even = single_fit["modeled_even_trace"]
single_model_odd = single_fit["modeled_odd_trace"]

single_rms_even = float(np.sqrt(np.mean((single_model_even - target_even_trace[::trace_fit_stride]) ** 2)))
single_rms_odd = float(np.sqrt(np.mean((single_model_odd - target_odd_trace[::trace_fit_stride]) ** 2)))

joint_results = []
for joint_half_span in joint_half_spans:
    selected_indices = np.arange(target_idx - joint_half_span, target_idx + joint_half_span + 1)
    selected_indices = np.clip(selected_indices, 0, useful_z.size - 1)
    selected_indices = np.unique(selected_indices)
    selected_local_idx = int(np.where(selected_indices == target_idx)[0][0])
    selected_offsets = selected_indices - selected_indices.min()
    selected_z = useful_z[selected_indices]

    chain_len = even_weights.size + int(np.max(selected_offsets))
    initial_phase_chain = build_initial_phase_chain(
        single_phase_seed,
        int(selected_offsets[selected_local_idx]),
        chain_len,
    )

    joint_even_traces = even_signal[:, selected_indices][::trace_fit_stride]
    joint_odd_traces = odd_signal[:, selected_indices][::trace_fit_stride]
    joint_even_delta_beta = even_delta_beta[::trace_fit_stride]
    joint_odd_delta_beta = odd_delta_beta[::trace_fit_stride]

    print(
        f"Starting joint nearby-coordinate uniqueness fit for {selected_indices.size} coordinates",
        flush=True,
    )
    joint_fit = fit_phase_chain_to_traces(
        even_weights,
        odd_weights,
        joint_even_traces,
        joint_odd_traces,
        joint_even_delta_beta,
        joint_odd_delta_beta,
        lag_step_m,
        selected_offsets,
        n_starts=trace_starts,
        random_seed=0,
        initial_amplitude_scale=single_fit["amplitude_scale"],
        initial_phase_chain=initial_phase_chain,
    )
    print(
        f"Joint nearby-coordinate fit finished for {selected_indices.size} coordinates",
        flush=True,
    )

    joint_phase_diff_mean, joint_phase_diff_std = phase_diff_stats_from_joint_solutions(
        joint_fit["all_solutions"],
        selected_local_idx,
    )
    joint_model_even_center = joint_fit["modeled_even_traces"][:, selected_local_idx]
    joint_model_odd_center = joint_fit["modeled_odd_traces"][:, selected_local_idx]
    joint_rms_even = float(
        np.sqrt(np.mean((joint_model_even_center - target_even_trace[::trace_fit_stride]) ** 2))
    )
    joint_rms_odd = float(
        np.sqrt(np.mean((joint_model_odd_center - target_odd_trace[::trace_fit_stride]) ** 2))
    )
    joint_results.append(
        {
            "coord_count": int(selected_indices.size),
            "selected_indices": selected_indices,
            "selected_z": selected_z,
            "selected_local_idx": selected_local_idx,
            "phase_diff_mean": joint_phase_diff_mean,
            "phase_diff_std": joint_phase_diff_std,
            "fit": joint_fit,
            "joint_even_traces": joint_even_traces,
            "joint_odd_traces": joint_odd_traces,
            "joint_even_delta_beta": joint_even_delta_beta,
            "joint_odd_delta_beta": joint_odd_delta_beta,
            "joint_model_even_center": joint_model_even_center,
            "joint_model_odd_center": joint_model_odd_center,
            "joint_rms_even": joint_rms_even,
            "joint_rms_odd": joint_rms_odd,
        }
    )

best_uniqueness = min(joint_results, key=lambda item: np.median(item["phase_diff_std"]))
largest_span_result = max(joint_results, key=lambda item: item["coord_count"])

print(f"File: {path_dat}")
print(f"Shape: {arr.shape}")
print(f"Target coordinate: z = {target_z:.3f} m")
print(
    "Nearby coordinates used: "
    + ", ".join(f"{z_val:.3f} m" for z_val in selected_z)
)
print(f"Pulse support size N: {even_weights.size}")
print(f"Discrete lag step: {lag_step_m:.6f} m")
print(f"Local window: traces {window_start} .. {window_end - 1} ({window_end - window_start} total)")
print(f"Local delta_beta span: {delta_beta_window[0]:.6f} .. {delta_beta_window[-1]:.6f} 1/m")
print(f"Harmonic starts: {harmonic_starts}")
print(f"Trace starts: {trace_starts}")
print(f"Trace fit stride: {trace_fit_stride}")
print(f"Single fit cost: {single_fit['cost']:.6g}")
print(f"Single RMS even/odd: {single_rms_even:.6g} / {single_rms_odd:.6g}")
print(
    f"Single phase-diff stability median/max: "
    f"{np.median(single_phase_diff_std):.6g} / {np.max(single_phase_diff_std):.6g} rad"
)
for result_item in joint_results:
    print(
        f"{result_item['coord_count']} coords fit cost: {result_item['fit']['cost']:.6g} | "
        f"RMS even/odd: {result_item['joint_rms_even']:.6g} / {result_item['joint_rms_odd']:.6g} | "
        f"phase-diff stability median/max: "
        f"{np.median(result_item['phase_diff_std']):.6g} / {np.max(result_item['phase_diff_std']):.6g} rad"
    )
print(
    f"Best uniqueness among joint fits: {best_uniqueness['coord_count']} coordinates "
    f"with median std {np.median(best_uniqueness['phase_diff_std']):.6g} rad"
)

plt.figure(figsize=(10, 4))
plt.plot(support_z, even_weights, lw=1.6, label="веса чётного импульса")
plt.plot(support_z, odd_weights, lw=1.6, label="веса нечётного импульса")
plt.xlabel("Coordinate z, m")
plt.ylabel("Weight A_m")
plt.title("Pulse weights used in uniqueness test")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(
    even_delta_beta[::trace_fit_stride],
    target_even_trace[::trace_fit_stride],
    lw=1.0,
    label="измерено",
)
axes[0].plot(
    even_delta_beta[::trace_fit_stride],
    single_model_even,
    lw=1.1,
    label="модель одной координаты",
)
for result_item in joint_results:
    axes[0].plot(
        even_delta_beta[::trace_fit_stride],
        result_item["joint_model_even_center"],
        lw=1.0,
        label=f"{result_item['coord_count']}-coord joint model",
    )
axes[0].set_ylabel("Чётный сигнал")
axes[0].set_title(f"Сравнение центральной трассы при z = {target_z:.3f} m")
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[1].plot(
    odd_delta_beta[::trace_fit_stride],
    target_odd_trace[::trace_fit_stride],
    lw=1.0,
    label="измерено",
)
axes[1].plot(
    odd_delta_beta[::trace_fit_stride],
    single_model_odd,
    lw=1.1,
    label="модель одной координаты",
)
for result_item in joint_results:
    axes[1].plot(
        odd_delta_beta[::trace_fit_stride],
        result_item["joint_model_odd_center"],
        lw=1.0,
        label=f"{result_item['coord_count']}-coord joint model",
    )
axes[1].set_xlabel(r"$\Delta \beta$, 1/m")
axes[1].set_ylabel("Нечётный сигнал")
axes[1].grid(True, alpha=0.3)
axes[1].legend()
fig.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(
    np.arange(1, even_weights.size),
    single_phase_diff_std,
    "o-",
    lw=1.3,
    ms=4,
    label="fit одной координаты",
)
markers = ["s-", "^-", "d-"]
for idx, result_item in enumerate(joint_results):
    plt.plot(
        np.arange(1, even_weights.size),
        result_item["phase_diff_std"],
        markers[idx % len(markers)],
        lw=1.3,
        ms=4,
        label=f"{result_item['coord_count']}-coord joint fit",
    )
plt.xlabel("Discrete lag index inside pulse")
plt.ylabel("Std of phase difference, rad")
plt.title("Phase-difference uniqueness comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig, axes = plt.subplots(
    2,
    largest_span_result["selected_indices"].size,
    figsize=(4 * largest_span_result["selected_indices"].size, 6),
    sharex=True,
)
for col, z_val in enumerate(largest_span_result["selected_z"]):
    axes[0, col].plot(
        largest_span_result["joint_even_delta_beta"],
        largest_span_result["joint_even_traces"][:, col],
        lw=1.0,
        label="измерено",
    )
    axes[0, col].plot(
        largest_span_result["joint_even_delta_beta"],
        largest_span_result["fit"]["modeled_even_traces"][:, col],
        lw=1.1,
        label="совместная модель",
    )
    axes[0, col].set_title(f"Чётные z = {z_val:.3f} m")
    axes[0, col].grid(True, alpha=0.3)
    axes[1, col].plot(
        largest_span_result["joint_odd_delta_beta"],
        largest_span_result["joint_odd_traces"][:, col],
        lw=1.0,
        label="измерено",
    )
    axes[1, col].plot(
        largest_span_result["joint_odd_delta_beta"],
        largest_span_result["fit"]["modeled_odd_traces"][:, col],
        lw=1.1,
        label="совместная модель",
    )
    axes[1, col].set_title(f"Нечётные z = {z_val:.3f} m")
    axes[1, col].grid(True, alpha=0.3)
    axes[1, col].set_xlabel(r"$\Delta \beta$, 1/m")
axes[0, 0].set_ylabel("Сигнал")
axes[1, 0].set_ylabel("Сигнал")
axes[0, 0].legend()
axes[1, 0].legend()
fig.tight_layout()

np.savetxt(
    save_dir / "uniqueness_comparison.csv",
    np.column_stack(
        [
            np.arange(1, even_weights.size, dtype=np.int64),
            single_phase_diff_mean,
            single_phase_diff_std,
        ]
        + [
            arr_col
            for result_item in joint_results
            for arr_col in (result_item["phase_diff_mean"], result_item["phase_diff_std"])
        ]
    ),
    delimiter=",",
    header=(
        "lag_index,"
        "single_mean_phase_difference_rad,single_std_phase_difference_rad,"
        + ",".join(
            f"joint_{result_item['coord_count']}_mean_phase_difference_rad,"
            f"joint_{result_item['coord_count']}_std_phase_difference_rad"
            for result_item in joint_results
        )
    ),
    comments="",
)

np.savetxt(
    save_dir / "nearby_coordinates.csv",
    np.column_stack(
        [
            largest_span_result["selected_indices"],
            largest_span_result["selected_indices"] - largest_span_result["selected_indices"].min(),
            largest_span_result["selected_z"],
        ]
    ),
    delimiter=",",
    header="useful_index,phase_chain_offset,z_m",
    comments="",
)

summary_rows = [[1, single_fit["cost"], single_rms_even, single_rms_odd, np.median(single_phase_diff_std), np.max(single_phase_diff_std)]]
for result_item in joint_results:
    summary_rows.append(
        [
            result_item["coord_count"],
            result_item["fit"]["cost"],
            result_item["joint_rms_even"],
            result_item["joint_rms_odd"],
            np.median(result_item["phase_diff_std"]),
            np.max(result_item["phase_diff_std"]),
        ]
    )
np.savetxt(
    save_dir / "joint_fit_summary.csv",
    np.asarray(summary_rows, dtype=np.float64),
    delimiter=",",
    header="coord_count,fit_cost,rms_even,rms_odd,median_phase_diff_std,max_phase_diff_std",
    comments="",
)

show_plots = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if show_plots:
    plt.show()
else:
    out = [
        save_dir / "phase_fit_pulse_weights.png",
        save_dir / "phase_fit_central_trace_comparison.png",
        save_dir / "phase_fit_uniqueness_comparison.png",
        save_dir / "phase_fit_nearby_traces.png",
    ]
    figs = [plt.figure(i) for i in plt.get_fignums()]
    for fig, path in zip(figs, out):
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print("Saved figures:")
    for path in out:
        print(path.resolve())
