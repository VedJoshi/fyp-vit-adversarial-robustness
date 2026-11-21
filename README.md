# Adversarial patch robustness in Vision Transformers

Final year project, BComp (Computer Science), National University of Singapore.

This project evaluates adversarial patch robustness in Vision Transformers. An adversarial patch
changes a bounded region of an image with the aim of causing misclassification. The main subject
of the audit is Robust Self-Attention (RSA), an inference-time defense that identifies anomalous
groups of image tokens and replaces their value vectors during attention.

The repository contains the evaluation code, attack implementations, reproducibility checks, and
machine-readable results. As of October 2026 the reproduction and gradient audit are complete,
and three inference-time changes to RSA's patch selection have been evaluated with attacks designed
against them. Two failed. The third, which replaces RSA's token-anomaly selector with an
input-space total-variation selector, is unsettled on development images: its pre-registered
comparison has not stabilised under added attack effort.

## Main result

RSA's released masking code repeats a `scatter` index 197 times. The repeated writes produce the
same forward output as a single write, but PyTorch sends gradient to every repeated source during
backpropagation. This amplifies the gradient path through the replacement mean without amplifying
the paths through the unmasked values.

Patch-AutoPGD therefore receives a substantially different gradient even though the defended
model produces bit-identical logits. The published robust accuracies reproduce under the
released-code derivative but not under the natural derivative implied by the algorithm.

The evaluation, numerical results, controls, and limits of this conclusion are in
[`01_gate_a/GATE_A.md`](01_gate_a/GATE_A.md).

## Repository structure

```text
code/fyp/          models, attacks, RSA, metrics, and instrumentation
code/scripts/      evaluation scripts and Slurm jobs
code/notebooks/    exploratory notebooks
code/results/      machine-readable result records
01_gate_a/         concise Gate A findings
refs.bib           project bibliography
```

## Setup

```bash
pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
cd code
python scripts/check_reference_fidelity.py
```

ImageNet data and model checkpoints are not included. See [`code/README.md`](code/README.md) for
the required data layout and experiment commands.

No license is currently granted.
