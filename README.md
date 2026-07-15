# Coherent Laser Reflectometer Analysis

Python scripts for processing coherent laser reflectometer `.dat` captures:

- baseline subtraction from a dark photodiode tail;
- even/odd reflectogram separation for alternating pulse lengths;
- colormaps of useful fiber regions with detected saw-sweep reset boundaries;
- pulse-shape extraction from the service zone;
- wavelength-sweep harmonic extraction;
- deconvolution of sweep harmonics into complex amplitudes and phases of discrete reflectors;
- post-sweep wavelength matching and piezo-driven phase tracking.

The current scripts were developed around captures where the first `0-110 m` are a service zone,
the useful fiber starts after that, and the final `50 m` can be used as a dark photodiode baseline.
Most command line arguments expose those assumptions so that other datasets can be processed without
editing code.

## Repository Contents

Core data handling:

- `raw_data.py` reads the binary `.dat` format and returns reflectograms as a 2D array.
- `reflectometer_utils.py` builds the distance axis and subtracts per-trace baseline from the dark tail.
- `analysis_output_utils.py` contains shared output naming helpers.

Visualization and preprocessing:

- `fiber_colormap_with_resets_even_odd.py` creates even/odd fiber colormaps with sweep reset lines.
- `pulse_profile_even_odd.py` extracts even/odd pulse profiles from the service zone.
- `match_reference_time_to_last_sweep.py` matches a post-modulation reference trace to the last wavelength sweep by direct correlation.

Sweep and harmonic analysis:

- `sweep_harmonics_even_odd.py` detects sweep boundaries and computes Fourier harmonics `H_p(z)`.
- `solve_pairwise_phase_differences.py` solves the local linear system that maps measured harmonics to pair products.
- `solve_complex_amplitudes_from_harmonics.py` performs the main deconvolution from `H_p(z)` to complex field samples `E_i`.

Piezo tracking:

- `track_two_discrete_piezo_phase.py` finds two adjacent discretes affected by the piezo and tracks their common phase over time.

Generated files such as `.dat`, `.mat`, `.png`, `.csv`, and `analysis_outputs/` are intentionally ignored by git.

## Environment

Tested with Python 3.11/3.13 and these packages:

```bash
python -m pip install -r requirements.txt
```

## Typical Workflow

Create colormaps for a dataset:

```bash
python fiber_colormap_with_resets_even_odd.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --baseline-tail-m 50 \
  --reset-period-override-ms 76.8 \
  --reset-anchor-time-s 0.0919 \
  --max-reset-time-s 4.45 \
  --shared-reset-detection
```

Extract pulse shapes:

```bash
python pulse_profile_even_odd.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --pulse-z-min 75 --pulse-z-max 85 \
  --zero-level-z 70 \
  --baseline-tail-m 50
```

Recover complex discrete amplitudes from the last sweep:

```bash
python solve_complex_amplitudes_from_harmonics.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --pulse-z-min 75 --pulse-z-max 85 --zero-level-z 70 \
  --sweep-span-pm 3.125 \
  --reset-period-override-ms 76.8 --reset-anchor-time-s 0.0919 \
  --shared-reset-detection --reset-detection-end-time-s 4.5 --max-reset-time-s 4.45 \
  --regularize-reset-grid --baseline-tail-m 50 \
  --sweep-index -1 --lag-min 2 --lag-max 16 \
  --direct-iters 20 --direct-damping 0.25 --direct-ridge-lambda 1e-3
```

Match a reference near `6 s` to the last sweep by direct correlation:

```bash
python match_reference_time_to_last_sweep.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --baseline-tail-m 50 \
  --reset-period-ms 76.8 --reset-anchor-time-s 0.0919 --max-reset-time-s 4.45 \
  --sweep-index -1 --sweep-span-pm 3.125 \
  --reference-time-s 6.0 \
  --reference-half-window-traces 32 \
  --sweep-half-window-traces 4
```

## Why `H_p(z)` Alone Does Not Directly Give Post-Sweep Wavelength Drift

During a saw sweep, the trace is sampled over a known wavelength interval. Fourier analysis over that
interval extracts harmonic coefficients `H_p(z)` that describe how interference terms with lag `p`
oscillate as wavelength changes. This is powerful while the sweep exists.

After modulation is switched off, each time point is only one intensity trace at one unknown wavelength.
That single trace is not enough to invert the full complex harmonic model uniquely:

- the measured intensity is real and loses absolute optical phase;
- phase is periodic modulo `2*pi`, so multiple wavelength shifts can produce similar patterns;
- piezo motion changes local phases independently of laser wavelength drift;
- noise and baseline errors can dominate a single trace;
- if the laser drifts outside the calibrated last-sweep span, the harmonic model is being extrapolated.

For that reason, post-sweep drift is better checked by independent methods:

- direct trace-bank matching against the last sweep;
- local-slope fitting only when the laser remains near the calibrated sweep point;
- using additional external wavelength/current/temperature readbacks when available.

See `docs/deconvolution_algorithm.md` for the full derivation of the harmonic deconvolution.
