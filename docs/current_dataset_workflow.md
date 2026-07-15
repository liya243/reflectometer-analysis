# Current Dataset Workflow

This file documents the parameter choices used for the current local dataset
`mem1_07_10-11_10_22.dat`.  The data file itself is not committed to git.

## Geometry and Timing Assumptions

- Scan rate: `10000 Hz`.
- Service zone: approximately `0-110 m`.
- Useful fiber zone used in most analyses: `110-360 m`.
- Dark photodiode baseline: final `50 m` of every trace.
- Pulse support used for deconvolution: `75-85 m`.
- Pulse zero-level coordinate: `70 m`.
- Alternating pulse lengths: even and odd reflectograms must be processed separately.
- Saw sweep span: `3.125 pm`.
- Accepted sweep period: `76.8 ms`.
- Accepted sweep-boundary anchor: `0.0919 s`.
- Last modulation boundary upper limit: `4.45 s`.
- Last full sweep used for deconvolution: `4.3159-4.3927 s`.

## Regenerate Colormaps

```bash
python fiber_colormap_with_resets_even_odd.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --baseline-tail-m 50 \
  --reset-period-override-ms 76.8 \
  --reset-anchor-time-s 0.0919 \
  --shared-reset-detection \
  --reset-detection-end-time-s 4.5 \
  --max-reset-time-s 4.45 \
  --regularize-reset-grid
```

## Regenerate Pulse Shapes

```bash
python pulse_profile_even_odd.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --pulse-z-min 75 --pulse-z-max 85 \
  --zero-level-z 70 \
  --baseline-tail-m 50
```

## Recover Discrete Complex Field from the Last Sweep

```bash
python solve_complex_amplitudes_from_harmonics.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --pulse-z-min 75 --pulse-z-max 85 --zero-level-z 70 \
  --sweep-span-pm 3.125 \
  --rolling-window 64 --min-period-s 0.03 --max-period-s 0.2 \
  --prominence-sigma 2.0 --refine-window-fraction 0.15 \
  --reset-time-shift-ms 0.0 \
  --reset-period-override-ms 76.8 --reset-anchor-time-s 0.0919 \
  --shared-reset-detection --reset-detection-end-time-s 4.5 --max-reset-time-s 4.45 \
  --regularize-reset-grid --baseline-tail-m 50 \
  --sweep-index -1 --lag-min 2 --lag-max 16 \
  --direct-iters 20 --direct-damping 0.25 --direct-ridge-lambda 1e-3
```

## Match a Post-Modulation Reference to the Last Sweep

The direct correlation check around `6 s` suggests the laser stopped near the middle of the last
calibrated sweep, roughly `1.6 pm` inside the `0..3.125 pm` sweep interval.  The correlation is weak,
so this is a sanity check rather than a precision wavelength measurement.

```bash
python match_reference_time_to_last_sweep.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --baseline-tail-m 50 \
  --reset-period-ms 76.8 --reset-anchor-time-s 0.0919 --max-reset-time-s 4.45 \
  --sweep-index -1 --sweep-span-pm 3.125 \
  --reference-time-s 6.0 \
  --reference-half-window-traces 32 \
  --sweep-half-window-traces 4 \
  --exclude-z-min 230 --exclude-z-max 240
```

## Track the Two-Discrete Piezo Phase

This analysis uses the complex field recovered from the last sweep and the currently selected
wavelength-drift correction.  The best candidate found in the current dataset is the adjacent pair:

- `230.446111 m`
- `230.965134 m`

```bash
python track_two_discrete_piezo_phase.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --baseline-tail-m 50 \
  --candidate-z-min 230 --candidate-z-max 240 \
  --phase-start-time-s 5.5 --baseline-duration-s 0.35 \
  --phase-grid-size 721 --block-size 32 \
  --fit-window-half-width-m 18 --rolling-window 9 \
  --lambda0-nm 1550 --drift-sign 1 \
  --phase-continuity-lambda 0.015 \
  --phase-zero-prior-lambda 0.05 \
  --parity-align-corr-floor 0.35
```
