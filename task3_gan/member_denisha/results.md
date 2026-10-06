# Task 3 results - Denisha Ketan Tank

## Model and training setup

This run trains a CycleGAN on two unpaired domains: 300 Monet images and 7,038
photo images. The implementation contains two generators and two PatchGAN
discriminators, adversarial least-squares loss, cycle-consistency L1 loss,
identity loss, replay buffers, gradient clipping, mixed-precision training, and
atomic resumable checkpoints.

| Setting | Value |
|---|---:|
| Image size | 256 x 256 |
| Batch size | 4 |
| Samples per epoch | 2,000 |
| Epochs | 60 |
| Decay start | Epoch 30 |
| Learning rate | 0.0002 |
| Cycle-loss weight | 10.0 |
| Identity-loss weight | 2.5 |
| Residual blocks | 9 |
| AMP | float16 |

## Training evidence

The archived GPU run used an NVIDIA GeForce RTX 4090, completed 60 epochs and
500 batches per epoch, and took 4,102.87 seconds. It exported 300 images in
each translation direction.

| Metric | Value |
|---|---:|
| Generator parameters | 22,756,358 |
| Discriminator parameters | 5,529,474 |
| Training time | 4,102.87 seconds |
| Image throughput | 58.50 images/sec |
| Peak GPU memory | 4,815.53 MB |
| Cycle L1, A | 0.077776 |
| Cycle L1, B | 0.091440 |
| Content cosine, A2B | 0.464679 |
| Content cosine, B2A | 0.304605 |
| Gradient NaN count | 0 |

The recorded generator-gradient mean/max fields are `NaN` in the archived
metrics file even though the explicit NaN counter is zero; this is reported as
a limitation rather than silently treated as a valid gradient statistic.

## Official evaluation and Kaggle submission

The supplied `Part3_Evaluation_Script.ipynb` was rerun on the 300-image cap in
both directions. It produced the submission committed at
`outputs/submission.csv`.

| Direction | FID | MiFID |
|---|---:|---:|
| Photo -> Monet (B2A) | 98.360 | 0.4069 |
| Monet -> Photo (A2B) | 103.228 | 0.4189 |
| Official average | 100.793810 | 0.412923 |

Kaggle record:

| Field | Value |
|---|---|
| Team | PairProgramming_Team_09 |
| Score | -50.6033 |
| Rank | 34 |
| Entries | 2 |
| Submission file | `outputs/submission.csv` |

## Human audit status

`outputs/human_audit.csv` contains the 30 fixed samples, but both raters' six
score columns are still blank. Cohen's kappa/percent agreement cannot be
computed honestly until two raters complete those fields. After completion run:

```bash
python src/audit_agreement.py
```

## Evidence map

- `src/train.py`: CycleGAN model, losses, training, inference, and checkpoints.
- `src/Part3_Evaluation_Script.ipynb`: supplied official FID/MiFID evaluator.
- `outputs/submission.csv`: official Kaggle-format submission.
- `outputs/submission_metrics_official.json`: official evaluation values.
- `outputs/history.json`: generator/discriminator/cycle/identity loss history.
- `outputs/metrics_A2B.json`, `outputs/metrics_B2A.json`: additional local metrics.
- `outputs/lpips_A.json`, `outputs/lpips_B.json`: reconstruction LPIPS.
- `outputs/human_audit.csv`: 30-sample audit template.
- `reproducibility/raw_logs/task3_denisha.log`: unedited GPU training log.
- `reproducibility/manifests/task3_denisha.json`: run manifest.
