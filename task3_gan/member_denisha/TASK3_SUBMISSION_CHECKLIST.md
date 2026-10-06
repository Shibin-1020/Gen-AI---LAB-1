# Task 3 submission checklist — Denisha Ketan Tank

## Automated requirement coverage

- Two unpaired image domains: approved `monet_jpg` and `photo_jpg` folders.
- CycleGAN architecture: two generators and two PatchGAN discriminators implemented in `src/train.py`.
- Adversarial, cycle-consistency, and identity losses: implemented and logged per epoch.
- GPU training: `--require-gpu` refuses to start if CUDA is unavailable.
- Smoke test: runs before full training and checks model forward/backward, finite values, and image shapes.
- Checkpointing: atomic `cyclegan_latest.pt` checkpoint every 100 batches plus epoch checkpoints. It contains both generators, both discriminators, all optimizer states, schedulers, epoch, config, and history.
- Resume: every full run passes `--resume-latest` and uses the same `checkpoints_competition` directory.
- Generated translations: 300 `pred_A2B` and 300 `pred_B2A` images are exported.
- Cycle verification: `recon_A`, `recon_B`, cycle L1, and content cosine are generated.
- Automated metrics: FID, KID, generative precision/recall, LPIPS, loss curves, gradient norms, NaN count, parameter counts, throughput, peak memory, and training time.
- Official submission protocol: `outputs/submission.csv` contains exactly `ID,FID,MiFID` with one result row, matching the supplied evaluation script.

## Run command

From the repository root in VSCode:

```bash
python task3_gan/member_denisha/run_task3.py \
  --config task3_gan/member_denisha/config_competition.yaml \
  --checkpoint-dir task3_gan/member_denisha/checkpoints_competition \
  --require-gpu
```

The final competition config is 60 epochs, batch size 4, 2,000 samples per epoch, and decay beginning at epoch 30.

## Manual requirements after training

1. Two raters must independently complete the six score columns for all 30 rows in `outputs/human_audit.csv`.
2. Run `python task3_gan/member_denisha/src/audit_agreement.py` and record the generated agreement metrics.
3. Upload `outputs/submission.csv` to Kaggle and record the leaderboard score and rank in the final report.

These three values cannot be generated honestly by the training code: the human scores require human raters, and the Kaggle score/rank requires the competition server.
