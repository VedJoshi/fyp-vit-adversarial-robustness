# Adversarial patch robustness in Vision Transformers

Final year project, BComp (Computer Science), National University of Singapore.

An adversarial patch is a small, visible region of an image an attacker is free to fill with
anything — a printed sticker on a road sign, say. Vision Transformers are particularly exposed to
them, because self-attention lets one token influence every other token directly, so a patch does
not have to corrupt the image to change the prediction. It only has to capture the attention map.

This project audits **RSA** (Robust Self-Attention; Mu & Wagner, ICML 2021 UDL Workshop), a defense
that runs at inference time. At each attention layer RSA scores every image token by how far its
value vector sits from the mean, then masks the highest-scoring window of tokens. The question is
whether a defense of that shape survives an attacker who knows it is there — the standard it has
not previously been held to.

The audit is in two parts. First reproduce the defense's published numbers, since a defense
evaluated against a faulty reimplementation tells you nothing. Then attack it adaptively.

## What is here

Code and run records only. The written work, reading notes and PDF library are kept separately.

```
code/
├── fyp/            the package: model, data, attacks, the RSA defense, instrumentation
├── scripts/        one entry point per experiment, plus Slurm job files
├── notebooks/      three notebooks, in order
└── results/        one JSON per run
```

`code/README.md` is the working guide: what each module does, how to run each experiment, the
measured cost of each, and the things that bite.

## Running it

```bash
pip install torch torchvision timm matplotlib jupyterlab
cd code
jupyter lab notebooks/
```

`config.SCALE` defaults to `FULL`, which expects ImageNet-100 on disk. Set it to `DEV` for a
one-minute version that downloads Imagenette instead.

## Status

The harness, defense, attacks and instrumentation are built and validated, each with a record in
`code/results/`.

The reproduction has run. **Its figures are withheld** until one control completes: the authors
evaluate weights produced on ImageNet-100, while this work uses an ImageNet-1k model with the
classifier head sliced to 100 classes, on a different 100-class draw. RSA decides what to mask
from value-vector geometry, so the checkpoint bears on the quantity being measured. The sweep and
the control are both in this repository; only the output is held back, and it goes in when the
measurement exists.

## Reuse

The RSA reimplementation is this project's own, verified against the authors' released code but
not derived from it. No licence is set; please ask before reusing.
