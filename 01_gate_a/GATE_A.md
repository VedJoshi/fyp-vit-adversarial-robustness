# Gate A: RSA reproduction findings

## Question

Can RSA's published adversarial patch robustness be reproduced?

## Evaluation

- Model: ImageNet-trained DeiT-small with a 100-class output head
- Dataset: 512 ImageNet-100 validation images
- Attack: 100-step Patch-AutoPGD with DLR loss
- Patch sizes: 10, 20, 30, 40, and 50 px
- Locations: worst case over the paper's stride-20 grid

Three defended evaluations were compared:

1. `natural`: the derivative implied by RSA's algorithm
2. `released derivative`: only the masking derivative changed to match the released code
3. `full released`: released derivative, AutoPGD schedule, and survivor filtering

The natural and released-derivative arms compute bit-identical masked values and logits.

## Results

| Patch | Natural | Released derivative | Full released | RSA Table 1 |
|---:|---:|---:|---:|---:|
| 10 px | 45.31% | 83.01% | 82.81% | 83.20% |
| 20 px | 0.00% | 74.02% | 74.61% | 72.46% |
| 30 px | 0.00% | 69.73% | 69.34% | 68.55% |
| 40 px | 0.00% | 60.16% | 61.33% | 56.25% |
| 50 px | 0.00% | 49.22% | 49.41% | 43.16% |

The derivative-only arm is within 1.17 percentage points of the full released arm at every size.
The derivative accounts for essentially the complete difference between the natural and released
evaluations.

## Cause

The released implementation contains this masking operation:

```python
v = v.scatter(-2, idx.repeat(1, H, context_length, D), v_mean)
```

`context_length` is 197. The operation writes the same mean vector to the same destination 197
times. Repeated identical writes leave the forward result unchanged. During backpropagation,
PyTorch routes gradient to every repeated source and sums the copies. This amplifies the path
through the mean vector without amplifying the paths through unmasked values.

On DeiT-small at 20 px, the natural and released input gradients have cosine similarity -0.005
and disagree on the sign of 50.3% of pixels.

## Controls

- The undefended evaluation reproduces RSA's undefended row within 1.37 percentage points.
- Clean accuracy is identical across the three defended derivative arms.
- Window rule, score frame, renormalisation, class draw, and checkpoint differences were tested or
  bounded separately.
- Gradient-masking warning-sign tests distinguish the released derivative from a normally
  functioning but difficult optimization landscape.

## Conclusion

The released-code derivative substantially weakens Patch-AutoPGD and exhibits gradient-masking
behavior. RSA's published results reproduce under this derivative but collapse under the natural
derivative in this evaluation.

This does not establish which evaluation path produced Table 1 in 2021, imply intent, or establish
RSA's robustness against every possible attack.

## Records

- `code/results/gate_a_reference.json`
- `code/results/gate_a_bwdrel.json`
- `code/results/gate_a_released.json`
- `code/results/gate_a_undefended.json`
- `code/results/m4_patch_warning_signs_intended.json`
- `code/results/m4_patch_warning_signs_released.json`
