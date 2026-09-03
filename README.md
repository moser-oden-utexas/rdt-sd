# rdt-sd

Generates rapidly distorted turbulence velocity spectra on spherical t-design
wavevector grids, either as an ensemble of sampled cases or a single case.



## Pipeline

| Stage | Module | What it does |
| --- | --- | --- |
| Sampling | `src/sampler.py` | Draws 7 parameters via a Sobol sequence — 4 shaping the mean velocity gradient (strain/rotation blend), 3 the Coriolis vector. Ensemble runs only; single-case runs supply `grad_u`/`omega` directly. |
| Grids | `src/spherical_designs.py` | Loads the spherical t-design wavevector grid. Checks `grids/` first; on a miss, downloads from UNSW and caches. |
| Solving | `src/rdt_solver.py` | Integrates the RDT ODE system with JAX/diffrax, using adaptive `dopri5` or fixed-step `rk4`. |
| Saving | `scripts/launcher.py` | Writes to `results/<run_name>/`: one `phi_batch_{i}.npy` per batch for ensembles, or a single `phi_single.npy`. |
| Postprocessing | `scripts/postprocessing.py` | Reads saved phi arrays, computes structure tensors (`src/tensor_utils.py`) and the early-stopping index per case (`src/earlystopping.py`), and writes a pickled tensors dict plus an `.npy` es_array. |
| ML dataset | `scripts/ml_postprocessing.py` | Normalizes the structure tensors by `q2`, truncates each case at its early-stopping index, flattens case and time into one sample axis, attaches the per-case velocity gradients and Coriolis terms, drops unrealizable samples, and plots the resulting anisotropy spread on the barycentric triangle. |
| Anisotropy visualization | `scripts/barycentric_plots_w_coriolis.py` | Maps the R_ij and D_ij anisotropies onto the barycentric (Lumley) triangle, truncating each case at its early-stopping index. |

## Usage

Run from the repo root.

### Simulating 

Edit `configs/launcher_config.toml`, then:

```
python -m scripts.launcher
```

### Postprocessing 

Point it at the saved phi arrays; multiple paths are treated as
batch shards of one ensemble and concatenated:

```
python -m scripts.postprocessing \
  --phi_arrays results/<run_name>/phi_batch_{0..NUM_BATCHES-1}.npy \
  --structure_tensors_output results/<run_name>/structure_tensors.pkl \
  --es_array_output results/<run_name>/es_array.npy \
  --es_degree et
```

`--es_threshold` defaults to `1.6e-4`. Add `--es_only` to skip the structure
tensor computation and only refresh es_array, which also drops the need for
`--structure_tensors_output`. `--es_degree` is based on the spherical design grid, and must be set to
`2*(t//4)`, the largest even integer at most `t/2`, where `t` is the spherical
design degree. The halving is because the coefficient is computed by quadrature
of the product `Y_lm * phi`, whose degree is `l + deg(phi)`, so a `t`-design is
only exact while `2l <= t`. The evenness is because `phi` is even in `k`, which
makes every odd-`l` coefficient vanish identically. NUM_BATCHES is the number of batches written,
`ceil(num_samples / batch_size)` from `launcher_config.toml`.

### Building the ML dataset

Point it at the postprocessed outputs and at the launcher config that generated
the run:

```
python -m scripts.ml_postprocessing \
  --structure_tensors results/<run_name>/structure_tensors.pkl \
  --es_array results/<run_name>/es_array.npy \
  --config configs/launcher_config.toml \
  --output_dir results/<run_name>/ml
```

Writes one `.npy` per array — `q2`, `r_ij`, `d_ij`, `q_ijk`, `qs_ijk`, `m_ijpq`,
`ms_ijpq`, `l_ijpq`, `j_ijrpq`, `gradU`, and `corU` when the run used Coriolis —
all sharing a trailing sample axis. The mean velocity gradients and Coriolis terms
are not saved by the launcher, so they are re-resolved from `--config` — re-sampled
from its seed, or reloaded from its `grad_u_location`/`coriolis_location`. It must be
the config that produced the run, or the cases will not line up.

Alongside the arrays it writes `anisotropy_plot_Rij.svg` and
`anisotropy_plot_Dij.svg`, showing the spread of the dataset's anisotropy states on
the barycentric (Lumley) triangle. Datasets run to tens of thousands of samples, so
only every `BARY_PLOTTING_FREQUENCY`-th sample is drawn; edit that module constant
to change the density.

### Building train/test ML datasets

`scripts/build_ml_datasets.py` drives the simulate → postprocess → build-dataset
pipeline above once per split to produce a training dataset and `n` testing
datasets, each integrated to a different `St_max`:

```
python -m scripts.build_ml_datasets \
  --config configs/launcher_config.toml \
  --st_train 3 \
  --st_test 3 4 5 \
  --train_num_cases 256 \
  --test_num_cases 32
```

`--train_num_cases + len(--st_test) * --test_num_cases` cases are sampled once
from `[ensemble] seed` in `--config`, then sliced into contiguous,
non-overlapping blocks — one per split — so train and every test dataset are
disjoint by construction. `num_time_steps` is scaled per split
(`round(num_time_steps * st_max / st_train)`) so the St resolution
(`st_max / num_time_steps`) matches training for every test dataset. `--config`
otherwise supplies `batch_size`, `use_coriolis`, `sd_degree`, and `solver`;
`--es_degree` defaults to `2 * (sd_degree // 4)` and `--es_threshold` to
`1.6e-4`, matching `scripts/postprocessing.py`.

Writes `results/<run_name>_train/` and one `results/<run_name>_test_st<St>/`
per test value, each with the same layout as a single `scripts.launcher` +
`scripts.postprocessing` + `scripts.ml_postprocessing` run, including its own
`ml/` dataset directory.

### Visualizing Anisotropy 


```
python -m scripts.barycentric_plots_w_coriolis
```

This reads cached ensembles from `results/runs/` for the fixed cases
`coriolis_rdt_jcp` and `no_coriolis_rdt_jcp`, expecting
`structure_tensors_<case>.pkl` and `<case>_es_array.npy`. Postprocessed outputs
must be placed there under those names before plotting.

## Config reference

`configs/launcher_config.toml` has the following args: 

- `[run]` — `name`, the output directory under `results/`.
- `[run_type]` — mode switches. `ensemble = true` resolves and solves a batch of
  cases via `simulate_ensemble_cases`; `stages = true` solves the one compound
  case in `[stages]` via `launch_rdt_stages`; with both `false`, the one case in
  `[single]` is solved via `launch_rdt_single`. Setting both `true` is an error.
- `[params]` — args shared by all modes: `num_time_steps`, `sd_degree`,
  `st_max`, and `solver` (`"dopri5"` or `"rk4"`). In `stages` mode `st_max` is
  ignored, since it comes per stage from `[stages]`.
- `[ensemble]` — ensemble-only args: `num_samples`, `batch_size`, `seed`, and
  `use_coriolis`. `use_coriolis` is the Coriolis switch: `true` samples Coriolis
  vectors and applies them, `false` runs without rotation. Set `[run] name` per
  case so the two ensembles land in separate directories. `case_offset` and
  `total_num_samples` are optional and only set by `scripts/build_ml_datasets.py`:
  when present, cases are sampled as `total_num_samples` and sliced down to
  `num_samples` starting at `case_offset`, instead of sampling `num_samples`
  cases from index `0`. This is useful for creating nice disjoint sampling spaces for train and test datasets, by sampling a sobol sequence once (instead of sampling twice) first and then partitioning into train and test parameters.
  `grad_u_location` and `coriolis_location` are optional and unset by default:
  when unset, cases come from the Sobol sequence as described above. When
  `grad_u_location` points at an `.npy` file of shape `(N, 3, 3)`, those mean
  velocity gradients are used instead, `num_samples` and `seed` are ignored, and
  the file length sets the number of cases — `scripts/launcher.py` records the
  resolved count as `num_samples` in the `config.toml` it saves. With
  `use_coriolis = true`, `coriolis_location` is required alongside it and must
  point at an `.npy` file of shape `(N, 3)` with the same `N`; with
  `use_coriolis = false`, `grad_u_location` may be given on its own and
  `coriolis_location` must be left unset.
- `[single]` — single-case-only args: `grad_u`, shape (3, 3), and `omega`,
  shape (3,), defaulting to zeros.
- `[stages]` — compound-distortion args, for one case driven by a
  piecewise-constant mean velocity gradient: `grad_u`, shape
  (num_stages, 3, 3), `st_max`, shape (num_stages,), and `omega`, shape (3,),
  shared by all stages and defaulting to zeros. Stage `i` applies `grad_u[i]`
  for `st_max[i]` of *its own* strain time — each stage derives its physical
  duration from `strain_rate(grad_u[i])` — starting from the state stage `i - 1`
  ended on. So `grad_u = [G1, G2]` with `st_max = [1.0, 1.0]` runs `G1` over
  `St = 0 -> 1` and `G2` over `St = 1 -> 2`, and the run spans `sum(st_max)` in
  total. `num_time_steps` is the total across all stages, allocated
  proportionally to each stage's `St` span, so snapshots land on one globally
  uniform `St` grid and the saved `phi_single.npy` has the same
  `(num_time_steps, 9, n_wavevectors)` layout as a `[single]` run —
  `scripts/postprocessing.py` consumes it unchanged. Alongside it the run
  directory gets `st_axis.npy`, shape (num_time_steps,), and `kappa_init.npy`,
  shape (3, n_wavevectors), holding the undeformed spherical-design wavevectors.

## Scripts

Supplementary figure and analysis scripts, each run as `python -m scripts.<name>`:

| Script | Purpose | Output |
| --- | --- | --- |
| `visualize_sd.py` | Renders t-design grids as 3D scatters on a unit sphere (`--degrees`, default `5 19 45`). | `results/spherical_designs/k_spherical_designs_t{t}.svg` |
| `convergence_plots.py` | Convergence against analytical solutions, with normalized time steps and with spherical t-design degree. | `results/convergence_plots/*.svg` |
| `earlystopping_thresholds.py` | Stopping thresholds from lmax energy fractions and M6 errors of Axisymmetric expansion. | `results/es_thresholds/*.svg` |
| `verify_es.py` | Validates the Stopping criterion on pure shear. | `results/verify_es/*.csv` |
| `push_to_1c.py` | Pushes anisotropy toward the 1C limit using frame rotation (`--regenerate` to rebuild the cache). | `results/push_1c/*.svg` |
| `build_ml_datasets.py` | Builds train + n test ML datasets spanning different St ranges from one disjoint sampled pool; see [Building train/test ML datasets](#building-traintest-ml-datasets). | `results/<run_name>_train/`, `results/<run_name>_test_st<St>/` |

## Anisotropy spreads

Two 200-case ensembles over the same sampled velocity gradients, run with
`use_coriolis` off and on, so the columns below isolate the effect of frame
rotation. Every dot is one realizable anisotropy state on the barycentric
(Lumley) triangle, and each case is truncated at its early-stopping index.

|  | Without Coriolis | With Coriolis |
| --- | --- | --- |
| **R_ij** (componentality, corners `1C`/`2C`/`3C`) | <img src="results/barycentric_plots/no_coriolis_rdt_jcp_anisotropy_plot_Rij.svg" width="330"> | <img src="results/barycentric_plots/coriolis_rdt_jcp_anisotropy_plot_Rij.svg" width="330"> |
| **D_ij** (dimensionality, corners `1D`/`2D`/`3D`) | <img src="results/barycentric_plots/no_coriolis_rdt_jcp_anisotropy_plot_Dij.svg" width="330"> | <img src="results/barycentric_plots/coriolis_rdt_jcp_anisotropy_plot_Dij.svg" width="330"> |

Regenerate with `python -m scripts.barycentric_plots_w_coriolis`, as described
under [Visualizing Anisotropy](#visualizing-anisotropy).
