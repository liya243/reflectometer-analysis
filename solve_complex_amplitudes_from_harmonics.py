"""Recover discrete complex reflector amplitudes from wavelength-sweep harmonics.

This is the central deconvolution script in the project.  It uses one calibrated
wavelength sweep to recover a complex discrete field

    E_i = A_i * exp(i phi_i)

along the fiber.  The algorithm is:

1. Split alternating even/odd reflectograms because the pulse lengths differ.
2. Detect or accept saw-sweep reset boundaries.
3. Fit Fourier harmonics H_p(z) over one selected wavelength sweep.
4. Use known even/odd pulse shapes to deconvolve H_p(z) into pair products
   X_i,p ~= E_i * conj(E_{i+p}).
5. Factor those products as a rank-1 Hermitian matrix E E^H.
6. Refine E by fitting the measured H_p(z) maps directly.

The recovered field is defined only up to a global phase.  The script fixes that
gauge by rotating the first complex sample to be real and positive.  See
docs/deconvolution_algorithm.md for the full derivation and limitations.
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
        raise ValueError("reset period override must be positive")

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
    """Flatten solved pair-product diagonals into weighted graph observations.

    After the linear harmonic deconvolution we know noisy estimates of

        X_i,p = E_i * conj(E_{i+p})

    for multiple lags p.  It is convenient to treat every valid entry as an
    edge in a graph: node i is connected to node j=i+p and the measured edge
    value is the complex product.  Later reconstruction asks for the node
    values E_i that best explain all these edges.

    Very small |X_i,p| values have nearly random phase, so `amplitude_floor`
    removes them before they can destabilize phase recovery.
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
        raise ValueError("No diagonal entries survive amplitude_floor")
    return {
        "i": np.concatenate(obs_i).astype(np.int64),
        "j": np.concatenate(obs_j).astype(np.int64),
        "value": np.concatenate(obs_value).astype(np.complex128),
        "weight": np.concatenate(obs_weight).astype(np.float64),
        "chain_length": int(chain_length),
    }


def solve_log_magnitudes(obs, chain_length, ridge_lambda, eps=1e-12):
    """Estimate |E_i| from the magnitudes of pair products.

    Because

        |E_i * conj(E_j)| = |E_i| * |E_j|,

    the logarithm turns the magnitude problem into a linear system:

        log |X_ij| = log |E_i| + log |E_j|.

    The extra final row is a weak anchor preventing an almost-singular graph
    from drifting by a common scale in poorly connected regions.
    """
    rows = obs["value"].size
    system = np.zeros((rows + 1, chain_length), dtype=np.float64)
    rhs = np.zeros(rows + 1, dtype=np.float64)
    weights = np.sqrt(np.maximum(obs["weight"], eps))
    for row, (i_idx, j_idx, value, row_weight) in enumerate(zip(obs["i"], obs["j"], obs["value"], weights)):
        system[row, i_idx] = row_weight
        system[row, j_idx] = row_weight
        rhs[row] = row_weight * np.log(max(abs(value), eps))
    # Mild anchor to avoid global drift when the graph is nearly singular.
    system[-1, 0] = np.sqrt(float(ridge_lambda) + eps)
    rhs[-1] = 0.0
    solution, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    return np.exp(solution)


def solve_recursive_phases(obs, chain_length):
    """Build a rough phase seed from nearest-neighbor pair products.

    If X_i,1 = E_i * conj(E_{i+1}), then

        arg(X_i,1) = phase(E_i) - phase(E_{i+1}).

    This gives phase(E_{i+1}) recursively.  Missing nearest-neighbor products
    are bridged by keeping the previous phase; the later ALS stages refine this
    rough seed using all available lags.
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
    """Factor noisy pair products into a rank-1 Hermitian field estimate.

    The ideal matrix of all pair products is

        X_ij = E_i * conj(E_j) = E E^H.

    Only some diagonals of this matrix are measured, and they are noisy.  This
    alternating least-squares loop updates one complex sample E_n at a time
    while keeping all other samples fixed.  For fixed neighbors the weighted
    least-squares update has a simple closed form.

    The result is used as a stable initialization for the later direct fit to
    measured harmonics, not as the final answer.
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
    """Forward model: predict H_p(z) from a candidate discrete field E.

    For each lag p and coordinate z_s, the harmonic is the valid convolution
    of the pair-product diagonal E_i*conj(E_{i+p}) with the pulse-overlap
    kernel weights[k] * weights[k+p].
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
    """Refine E by fitting the measured even/odd harmonic maps directly.

    The two-stage reconstruction `H_p -> pair products -> E` can amplify noise,
    especially for weak lags.  This function therefore returns to the original
    measured quantities and minimizes the mismatch between measured H_p(z) and
    the harmonics predicted by the current E.

    The update is coordinate-wise.  When all field samples except E_n are fixed,
    every affected harmonic residual is linear in Re(E_n) and Im(E_n).  Each
    update is therefore a small real least-squares solve with ridge
    regularization, followed by damping to avoid large unstable jumps.
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
        # Normalize every lag before stacking residuals.  Without this, strong
        # low-order harmonics dominate the cost and weak but informative lags
        # have almost no influence on the fit.
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
                    # H_p(z_s) contains terms E_{s+m}*conj(E_{s+m+p}).
                    # E_n can appear either as the left member of a pair
                    # (n=s+m) or as the right member (n=s+m+p).  The candidate
                    # coordinates below are exactly those affected rows s.
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

                        # Remove the old contribution of E_n from the current
                        # prediction, then solve for the E_n that would best
                        # explain the remaining target.  The complex expression
                        # alpha*E_n + beta*conj(E_n) is linear in Re(E_n) and
                        # Im(E_n), which gives the two real design columns below.
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
        "message": "ALS direct fit completed",
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

f1 = figure('Color', 'w', 'Name', 'Recovered complex amplitudes');
subplot(2,1,1);
plot(data.chain_distance_m, data.E_amplitude, 'LineWidth', 1.4);
grid on; xlabel('Distance (m)'); ylabel('|E|'); title('Recovered complex amplitude magnitude');
subplot(2,1,2);
plot(data.chain_distance_m, data.E_phase, 'LineWidth', 1.4);
grid on; xlabel('Distance (m)'); ylabel('arg(E) (rad)'); title('Recovered complex amplitude phase');

f2 = figure('Color', 'w', 'Name', 'Recovered pairwise products');
subplot(2,1,1);
imagesc(data.lag_indices, data.chain_distance_m, data.reconstructed_product_phase);
axis xy; colorbar; xlabel('Lag p'); ylabel('Distance (m)'); title('arg(E_i conj(E_{{i+p}}))');
subplot(2,1,2);
imagesc(data.lag_indices, data.chain_distance_m, data.reconstructed_product_amplitude);
axis xy; colorbar; xlabel('Lag p'); ylabel('Distance (m)'); title('|E_i conj(E_{{i+p}})|');
"""
    script_path.write_text(script_text, encoding="utf-8")
    return {"mat": mat_path, "script": script_path}


def main():
    parser = argparse.ArgumentParser(
        description="Recover a complex discrete field E_i from one sweep of H_p(z) via X = E E^H factorization."
    )
    parser.add_argument("dat_path", help="Path to the .dat file")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Directory for output files")
    parser.add_argument("--scan-rate", type=float, default=None, help="Optional override for reflectogram scan rate in Hz")
    parser.add_argument("--fiber-z-min", type=float, default=105.0, help="Start of real fiber region in meters")
    parser.add_argument("--fiber-z-max", type=float, default=280.0, help="End of real fiber region in meters")
    parser.add_argument("--pulse-z-min", type=float, default=75.0, help="Start of pulse support in meters")
    parser.add_argument("--pulse-z-max", type=float, default=85.0, help="End of pulse support in meters")
    parser.add_argument("--zero-level-z", type=float, default=70.0, help="Zero level for pulse weights in meters")
    parser.add_argument("--lambda0-nm", type=float, default=1550.0, help="Central wavelength in nm")
    parser.add_argument("--sweep-span-pm", type=float, default=3.125, help="Wavelength span of one sweep in pm")
    parser.add_argument("--rolling-window", type=int, default=64, help="Reset detector smoothing window in traces")
    parser.add_argument("--min-period-s", type=float, default=0.05, help="Minimum sweep period for reset detection")
    parser.add_argument("--max-period-s", type=float, default=None, help="Maximum sweep period for reset detection")
    parser.add_argument("--prominence-sigma", type=float, default=2.0, help="Reset detector threshold in robust sigma units")
    parser.add_argument("--refine-window-fraction", type=float, default=0.15, help="Local refinement window as fraction of detected period")
    parser.add_argument("--reset-time-shift-ms", type=float, default=3.0, help="Shift detected sweep-end times later by this many milliseconds")
    parser.add_argument("--reset-period-override-ms", type=float, default=None, help="Use this fixed reset period in milliseconds for sweep boundaries")
    parser.add_argument("--reset-anchor-time-s", type=float, default=None, help="Anchor time for fixed reset grid; grid is extended backward and forward")
    parser.add_argument("--shared-reset-detection", action="store_true", help="Use one common reset grid for even and odd traces")
    parser.add_argument("--max-reset-time-s", type=float, default=None, help="Keep only detected reset times not later than this time")
    parser.add_argument("--reset-detection-end-time-s", type=float, default=None, help="Use only traces up to this time when detecting reset grid")
    parser.add_argument("--regularize-reset-grid", action="store_true", help="Project detected reset times onto a regular grid with the fitted period")
    parser.add_argument("--baseline-tail-m", type=float, default=50.0, help="Subtract per-trace baseline estimated from the last this many meters")
    parser.add_argument("--ridge-lambda", type=float, default=1e-6, help="Ridge regularization for diagonal least squares")
    parser.add_argument("--amplitude-floor", type=float, default=1e-4, help="Ignore solved pair products below this floor")
    parser.add_argument("--als-iters", type=int, default=20, help="Alternating minimization iterations for E recovery")
    parser.add_argument("--sweep-index", type=int, default=0, help="Zero-based sweep index to use instead of averaging over sweeps")
    parser.add_argument("--lag-min", type=int, default=1, help="Minimum lag p to use in the direct nonlinear fit")
    parser.add_argument("--lag-max", type=int, default=None, help="Maximum lag p to use in the direct nonlinear fit")
    parser.add_argument("--direct-iters", type=int, default=20, help="ALS iterations for direct fit to measured harmonics")
    parser.add_argument("--direct-damping", type=float, default=0.25, help="Damping factor for direct ALS updates")
    parser.add_argument("--direct-ridge-lambda", type=float, default=1e-3, help="Ridge regularization for each scalar ALS update")
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
                    raise ValueError("Cannot infer reset anchor from an empty reset list")
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
                        raise ValueError("Cannot infer reset anchor from an empty reset list")
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
                f"sweep_index {args.sweep_index} is out of range for parity '{parity}' with {harmonic_cube.shape[0]} complete sweeps"
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
        raise ValueError("Selected lag window is empty")
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
    axes1[0].set_title("Recovered complex discrete amplitudes")
    axes1[0].grid(alpha=0.25)
    axes1[1].plot(chain_distance_m, e_phase, color="#111111", linewidth=1.4)
    axes1[1].set_xlabel("Distance (m)")
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
    axes2[0].set_ylabel("Distance (m)")
    axes2[0].set_title("Recovered pair products: arg(E_i conj(E_{i+p}))")
    fig2.colorbar(im0, ax=axes2[0], label="Phase (rad)")
    im1 = axes2[1].imshow(
        reconstructed_product_amplitude,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[lag_indices[0], lag_indices[-1], chain_distance_m[0], chain_distance_m[-1]],
    )
    axes2[1].set_xlabel("Lag p")
    axes2[1].set_ylabel("Distance (m)")
    axes2[1].set_title("Recovered pair products: |E_i conj(E_{i+p})|")
    fig2.colorbar(im1, ax=axes2[1], label="Amplitude")
    products_png_path = output_dir / f"{dat_path.stem}_recovered_pair_products.png"
    fig2.savefig(products_png_path, dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax3.plot(lag_indices, solved["residual_rms"], linewidth=1.5, label="Linear diagonal solve RMS")
    ax3.plot(lag_indices, factorization_error, linewidth=1.5, label="Rank-1 factorization mismatch")
    ax3.axvspan(lag_indices[0], int(args.lag_min), color="#DDDDDD", alpha=0.35, linewidth=0)
    if lag_max < int(lag_indices[-1]):
        ax3.axvspan(lag_max, lag_indices[-1], color="#DDDDDD", alpha=0.35, linewidth=0)
    ax3.set_xlabel("Lag p")
    ax3.set_ylabel("Error")
    ax3.set_title("Model mismatch by lag")
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
