# Evaluation code

This directory contains the implementation used to evaluate adversarial patch attacks and Robust
Self-Attention (RSA) on DeiT-small.

## Requirements

- Python 3.13
- PyTorch 2.7.0
- torchvision 0.22.0
- timm 1.0.15
- A CUDA-capable GPU for full evaluations

Install the pinned environment from the repository root:

```bash
pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

The result records in `results/` were generated with the pinned PyTorch and timm versions.

## Data

The full evaluation uses an ImageNet-100 validation set arranged as ImageFolder directories:

```text
<dataset-root>/val/n01440764/*.JPEG
<dataset-root>/val/<wnid>/*.JPEG
```

Set the dataset location before running an evaluation:

```bash
export FYP_IMAGENET100=/path/to/imagenet100
```

In PowerShell:

```powershell
$env:FYP_IMAGENET100 = "C:\path\to\imagenet100"
```

The expected class lists are stored in `data/`. The dataset can also be constructed from a source
directory or archive:

```bash
cd code
python scripts/build_imagenet100.py --help
```

ImageNet data and the released RSA checkpoint are not included in the repository.

## Configuration

`fyp/config.py` defines the model, image and token dimensions, random seeds, paths, and two run
scales:

| Scale | Dataset | Images | Attack steps | Purpose |
|---|---|---:|---:|---|
| `DEV` | Imagenette | 64 | 10 | local checks |
| `FULL` | ImageNet-100 | 512 | 100 | reported evaluations |

`SCALE` is currently set to `FULL`. Results produced with `DEV` are not comparable with the RSA
evaluation.

The following environment variables override the default paths:

| Variable | Purpose |
|---|---|
| `FYP_IMAGENET100` | ImageNet-100 root containing `val/` |
| `FYP_DATA_ROOT` | general data directory |
| `FYP_RESULTS_ROOT` | result output directory |

## Package structure

| Path | Contents |
|---|---|
| `fyp/config.py` | paths, model constants, scales, and seeding |
| `fyp/data.py` | datasets, class subsets, sampling, and data loaders |
| `fyp/models.py` | model loading, head slicing, normalization, and checkpoint loading |
| `fyp/attacks.py` | FGSM, PGD, patch PGD, Patch-AutoPGD, and Patch-Fool |
| `fyp/rsa.py` | RSA attention replacement and configuration |
| `fyp/metrics.py` | attention and robustness metrics |
| `fyp/hooks.py` | attention capture and reconstruction checks |
| `fyp/diagnostics.py` | attack diagnostics and gradient-masking checks |
| `fyp/results.py` | result serialization and provenance metadata |
| `scripts/` | evaluation and validation entry points |
| `scripts/slurm/` | cluster job scripts |
| `notebooks/` | exploratory baseline, attack, and instrumentation work |
| `results/` | machine-readable output records |

Images passed to the model and attacks are in `[0, 1]` pixel space. ImageNet normalization is
applied inside `NormalizedModel`. Model loading disables timm fused attention because the
instrumentation and RSA implementation require explicit attention tensors.

## Validation

Run the reference-fidelity checks after setup or after changing model, attack, or RSA code:

```bash
cd code
python scripts/check_reference_fidelity.py
```

The check covers attention reconstruction, RSA forward behavior, gradient behavior, attack
schedule behavior, location generation, and class-subset consistency.

The following commands provide additional targeted checks:

```bash
python scripts/warning_signs.py --help
python scripts/validate_patch_fool.py --help
python scripts/checkpoint_control.py --help
```

## Gate A evaluation

Gate A evaluates worst-case robust accuracy over the specified patch-location grid. The default
configuration uses the five RSA patch sizes, 512 images, and 100 Patch-AutoPGD steps.

```bash
cd code
python scripts/gate_a.py --tag full
```

A smaller run can be used to check the pipeline:

```bash
python scripts/gate_a.py --sizes 20 --n-images 128 --loc-stride 3 --tag pilot20
```

`--loc-stride` evaluates fewer locations and therefore does not produce a full worst-case result.
Use `python scripts/gate_a.py --help` for the complete set of protocol controls.

Gate A appends progress to `results/gate_a_<tag>.progress.jsonl` and resumes compatible interrupted
runs. Completed results are written to JSON. Existing headline records are:

- `results/gate_a_reference.json`
- `results/gate_a_bwdrel.json`
- `results/gate_a_released.json`
- `results/gate_a_undefended.json`

The findings and comparison table are in [`../01_gate_a/GATE_A.md`](../01_gate_a/GATE_A.md).

## Other evaluations

```bash
python scripts/table2_by_rule.py --help
python scripts/mechanism_full.py --help
python scripts/aas_efficacy.py --help
python scripts/logit_gap_by_model.py --help
python scripts/compare_runs.py --help
python scripts/merge_shards.py --help
```

The notebooks cover the baseline model, attack implementations, and attention instrumentation:

- `notebooks/M1_baseline.ipynb`
- `notebooks/M2_attacks.ipynb`
- `notebooks/M3_instrumentation.ipynb`

## Results

Result JSON files include model, dataset, sample size, attack settings, software versions, device,
Git commit, and working-tree state where available. Full evaluation results and partial validation
runs are distinguished in their `_meta` records.

Datasets, checkpoints, figures, temporary progress files, and cluster logs are excluded from Git.
