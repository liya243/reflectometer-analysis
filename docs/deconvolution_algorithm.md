# Deconvolution of Reflectometer Sweep Harmonics into Discrete Complex Fields

This document explains the main inverse problem implemented in
`solve_complex_amplitudes_from_harmonics.py`.

## Measurement Model

For one reflectogram coordinate `z`, the detected photodiode signal is a coherent sum of delayed
copies of the launched pulse. The service zone gives the pulse envelope sampled on the same distance
grid. Let:

- `E_i = A_i exp(i phi_i)` be the complex field contribution from discrete reflector `i`;
- `w_k` be the real pulse amplitude at discrete pulse sample `k`;
- `D` be one distance-grid step;
- `p` be a lag in discrete samples;
- `lambda` be laser wavelength;
- `beta(lambda) = 2*pi*n_eff/lambda`.

When two pulse samples separated by `p` overlap at coordinate `z`, the interference term contains

```text
E_i * conj(E_{i+p}) * exp(2 i beta(lambda) p D)
```

up to the known pulse weights. Therefore each lag `p` oscillates with a wavelength-dependent
frequency proportional to `pD`.

## Extracting Harmonics `H_p(z)`

During one saw-like wavelength sweep, `lambda` moves across a known span, e.g. `3.125 pm`.
For each coordinate `z`, the script fits the trace over that sweep to sinusoidal basis functions:

```text
cos(2 * delta_beta * pD), sin(2 * delta_beta * pD)
```

for all relevant lags `p`.

The fitted complex coefficient is called `H_p(z)`. Physically it is the weighted sum of all reflector
pairs separated by `p` that can contribute to coordinate `z`:

```text
H_p(z_s) = sum_k w_k * w_{k+p} * E_{s+k} * conj(E_{s+k+p})
```

Here `s` indexes the observed coordinate and `k` indexes the pulse support. This is a convolution:
for every lag `p`, the unknown pair-product diagonal

```text
X_i,p = E_i * conj(E_{i+p})
```

is blurred by a known kernel

```text
K_p,k = w_k * w_{k+p}
```

so that

```text
H_p = K_p (*) X_p
```

where `(*)` denotes a 1D valid convolution along the fiber coordinate.

## Step 1: Solve Pair-Product Diagonals

`solve_pairwise_phase_differences.py` solves the linear deconvolution problem separately for every
lag `p`.

Because the experiment alternates two pulse lengths, there are two measured harmonic sets:

```text
H_p_even(z)
H_p_odd(z)
```

and two known pulse kernels:

```text
K_p_even
K_p_odd
```

The same unknown pair-product diagonal `X_p` must explain both. The linear least-squares system is:

```text
K_p_even (*) X_p ~= H_p_even
K_p_odd  (*) X_p ~= H_p_odd
```

The solver uses ridge regularization because some lags have weak kernels or poor signal-to-noise:

```text
min_X ||A X - b||^2 + ridge_lambda * ||X||^2
```

The result is a set of complex diagonals:

```text
X_i,p ~= E_i * conj(E_{i+p})
```

These are not yet the field `E_i`; they are pairwise products.

## Step 2: Interpret the Diagonals as a Rank-1 Hermitian Matrix

If all pair products were known perfectly, they would form entries of a Hermitian rank-1 matrix:

```text
X_ij = E_i * conj(E_j)
```

The matrix has one unavoidable ambiguity:

```text
E_i -> E_i * exp(i theta_global)
```

does not change any product `E_i * conj(E_j)`. The scripts fix this by making the first recovered
sample real and positive.

In practice we only know a band of this matrix, because only finite lags are measurable and reliable.
The reconstruction is therefore approximate.

## Step 3: Initial Field Estimate

`solve_complex_amplitudes_from_harmonics.py` builds an observation list:

```text
i, j=i+p, measured_value = X_i,p
```

Weak pair products are dropped using `--amplitude-floor`, because their phase is mostly noise.

The initial magnitude estimate uses:

```text
log |X_i,p| = log |E_i| + log |E_{i+p}|
```

which is a linear least-squares problem for `log |E_i|`.

The initial phase estimate uses nearest-neighbor products when available:

```text
arg(E_i * conj(E_{i+1})) = phi_i - phi_{i+1}
```

so

```text
phi_{i+1} = phi_i - arg(X_i,1)
```

This gives only a rough seed.

## Step 4: Alternating Rank-1 Factorization

The script then improves the field by alternating minimization of:

```text
sum_observations weight_ij * |E_i * conj(E_j) - X_ij|^2
```

When all other `E_j` are fixed, the best update for one `E_i` has a closed form. This is implemented
by `alternating_rank1_hermitian`. It is not a full global optimizer, but it is stable enough to produce
a useful seed for the next stage.

## Step 5: Direct Fit Back to Measured Harmonics

The pair-product deconvolution step can amplify noise. Therefore the final stage fits `E_i` directly
against the measured harmonics again:

```text
predicted_H_p(z_s; E) =
    sum_k w_k * w_{k+p} * E_{s+k} * conj(E_{s+k+p})
```

for even and odd pulses simultaneously.

The direct fit minimizes normalized residuals between measured and predicted harmonic maps. It updates
one complex `E_i` at a time. For a single updated sample, the residual is linear in:

```text
Re(E_i), Im(E_i)
```

so every scalar update is a small real least-squares problem with ridge regularization:

```text
min ||A [Re(E_i), Im(E_i)] - b||^2 + direct_ridge_lambda * ||...||^2
```

`--direct-damping` mixes the update with the previous value to avoid unstable jumps:

```text
E_i <- (1 - damping) * E_i_old + damping * E_i_new
```

After every iteration the global phase is canonicalized again.

## Why Lag Selection Matters

Very small lags can be contaminated by baseline terms, pulse autocorrelation leakage, and imperfect
subtraction of the photodiode DC level. Very large lags have small pulse-kernel overlap:

```text
K_p,k = w_k * w_{k+p}
```

so they are weak and noisy. In practice `--lag-min` and `--lag-max` are physical filters:

- `lag-min` removes terms dominated by low-frequency/baseline artifacts;
- `lag-max` removes terms whose pulse overlap is too weak to recover reliably.

For the current dataset, a range like `--lag-min 2 --lag-max 16` was used as a pragmatic compromise.

## Why Post-Sweep Drift Is Hard to Recover from `H_p`

`H_p(z)` is calibrated during a known wavelength sweep. After modulation stops, each time point is a
single real-valued trace. Fitting that single trace against a complex harmonic model has several
degeneracies:

- phase is periodic modulo `2*pi`;
- multiple wavelength shifts can look similar if the trace is noisy;
- local piezo phase changes can mimic or obscure wavelength drift;
- outside the last sweep span, the model extrapolates rather than interpolates.

Therefore `H_p(z)` is excellent for reconstructing `E_i` from a sweep, but not by itself a reliable
absolute wavelength meter after the sweep is switched off.

## Main Quality Diagnostics

The deconvolution script saves:

- recovered `|E_i|` and unwrapped `arg(E_i)`;
- reconstructed pair-product amplitude and phase maps;
- model error by lag;
- MATLAB `.mat` data and opener scripts for interactive inspection.

Important printed metrics:

- `linear_residual_rms_mean`: quality of the linear `H_p -> X_p` deconvolution;
- `factorization_error_mean`: mismatch between solved pair products and `E_i conj(E_j)`;
- `direct_fit_residual_rms`: final mismatch between measured and modelled harmonics.
