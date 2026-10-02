# Task 3 — CycleGAN image style transfer (Monet ↔ Photo) · Shibin Thomas

Unpaired image-to-image translation between **A = Monet paintings** (300) and **B = photos** (7,038) with a CycleGAN
implemented and trained **from scratch**. Domain naming follows the instructor's evaluation script:
* `pred_A2B` = Monet → Photo (generated photos);
* `pred_B2A` = Photo → Monet (generated Monets).

| | |
|---|---|
| Config | [`configs/cyclegan_v1.yaml`](configs/cyclegan_v1.yaml) |
| Code / notebook | [`src/`](src/) · [`src/task3_cyclegan.ipynb`](src/task3_cyclegan.ipynb) · [`evaluate_local.py`](evaluate_local.py) |
| Predictions | [`outputs/pred_A2B/`](outputs/pred_A2B/), [`outputs/pred_B2A/`](outputs/pred_B2A/) (run id in `outputs/predictions_info.json`) |
| Kaggle file | [`submission.csv`](submission.csv) · leaderboard: [`kaggle_leaderboard.json`](kaggle_leaderboard.json) |
| All metrics | [`full_metrics_report.csv`](full_metrics_report.csv) (= `metrics_report.csv`) |
| Plots | `outputs/<run_id>/` loss_curves, epoch_curves, final_samples_*, failure_*, samples/epoch_*.png |
| Human audit | `outputs/human_audit/` (panels S01–S30, rater sheets, results) |
| Raw logs / manifest | `reproducibility/raw_logs/task3_gan/shibin_thomas/<run_id>/`, `reproducibility/manifests/task3_gan/shibin_thomas/<run_id>.json` |

## 1. Data: two unpaired domains (3.1.1)

| Domain | Folder | Images | Size |
|---|---|---|---|
| A — Monet paintings | `task3_gan/data/monet_jpg` | 300 | 256×256 RGB |
| B — photos | `task3_gan/data/photo_jpg` | 7,038 | 256×256 RGB |

* The class dataset zip also contained macOS metadata (`__MACOSX/`, `.DS_Store`). These were removed: they
  are not images and would break the image loaders.
* The images are not committed: the repository is public. Their file list is fingerprinted with SHA-256 in
  the run manifest.
* **Unpaired sampling.** Each step draws an independent Monet and photo; there is no correspondence
  between them. Monets are drawn from shuffled permutations, so all 300 paintings are used equally often;
  photos are drawn from shuffled permutations of all 7,038. One epoch = 2,000 pairs.
* **Augmentation** (CycleGAN "resize and crop"): bicubic resize to 286, random 256×256 crop, random
  horizontal flip; pixels scaled to [−1, 1] to match the generators' tanh output.
* **Imbalance (300 vs 7,038).** This is the main data challenge. Only 300 Monets define the target style,
  so D_A sees each painting about 267 times; augmentation and the image pool reduce memorisation.

## 2. Architecture (3.1.2) — two generators, two discriminators

| Network | Structure | Parameters |
|---|---|---|
| `G_AB` (Monet→Photo), `G_BA` (Photo→Monet) | c7s1-64 → d128 → d256 → **9 residual blocks** (256 ch, at 64×64) → up128 → up64 → c7s1-3 → tanh; InstanceNorm, ReLU, reflection padding | ≈ 11.4 M each |
| `D_A` (real/fake Monet), `D_B` (real/fake photo) | **70×70 PatchGAN**: C64 – C128 – C256 – C512 – 1 (4×4 convs, LeakyReLU 0.2, InstanceNorm except on the first layer); 30×30 output map for a 256×256 input | ≈ 2.77 M each |

Exact counts are in `full_metrics_report.csv`.

**Design choices and why:**
* **ResNet generator with 9 residual blocks** (Johnson et al. 2016), the standard choice for 256×256
  CycleGAN. Style transfer must keep the scene layout while changing texture and colour. Residual blocks
  carry the content through and let each block learn a correction, which suits "same scene, different style".
* **Resize-convolution upsampling** (nearest-neighbour ×2, then a 3×3 conv) instead of the paper's
  transposed convolutions. Transposed convolutions with stride 2 overlap unevenly and produce checkerboard
  artifacts (Odena et al. 2016), which are very visible in flat sky and water regions. This is my main
  architectural difference from the reference implementation; `model.upsample: deconv` restores the
  original.
* **InstanceNorm instead of BatchNorm.** Normalising each image separately removes image-specific contrast,
  which is the right inductive bias for style transfer (Ulyanov et al. 2016). It also works with small
  batches (4).
* **Reflection padding** avoids dark border artifacts at image edges.
* **70×70 PatchGAN.** It judges local texture (brush strokes, colour patches) rather than global layout,
  with far fewer parameters than a full-image discriminator; the cycle loss takes care of global structure.
* **Initialisation:** all weights N(0, 0.02), no pretrained weights.

## 3. Losses and training (3.1.3)

With a = real Monet and b = real photo:

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| Adversarial (LSGAN), generators | (D_B(G_AB(a)) − 1)² + (D_A(G_BA(b)) − 1)² | 1 | make translations look like the target domain |
| Cycle consistency | ‖G_BA(G_AB(a)) − a‖₁ + ‖G_AB(G_BA(b)) − b‖₁ | λ_cyc = 10 | translation must be invertible, so content is preserved (no mode collapse to one image) |
| Identity | ‖G_BA(a) − a‖₁ + ‖G_AB(b) − b‖₁ | λ_id = 5 (= 0.5 λ_cyc) | an image already in the target domain should not change; preserves colour composition (used by Zhu et al. for paintings) |
| Discriminators (LSGAN) | ½[(D(real) − 1)² + D(pool(fake))²] | — | D_A and D_B trained against a 50-image history pool of fakes |

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimiser | Adam, lr 2e-4, β = (0.5, 0.999), for G and D | the CycleGAN / DCGAN setting; β1 = 0.5 damps momentum oscillations in adversarial training |
| LR schedule | constant for 20 epochs, then linear decay to 0 over 20 epochs | as in Zhu et al.; decay lets the adversarial game settle |
| Epochs × pairs | 40 × 2,000 = 80,000 unpaired pairs (20,000 steps of batch 4) | each Monet is seen about 267 times, each photo about 11 times; enough for the style to converge within the GPU budget |
| Batch size | 4 | better GPU use than the paper's batch of 1; InstanceNorm makes the results independent of batch statistics |
| LSGAN instead of BCE | — | smoother gradients for samples far from the decision boundary, more stable training (Mao et al. 2017) |
| Image pool | 50 | discriminators also see older fakes, which reduces generator/discriminator oscillation (Shrivastava et al. 2017) |
| Precision | fp32 with TF32 matmuls | GAN losses are sensitive to bf16 rounding; TF32 is fast on the RTX 5090 and keeps fp32 range |
| Monitoring | every step: all loss terms, gradient norms of G and D, NaN check; every epoch: cycle L1 on a fixed batch, sample grid; every 5 epochs: FID in both directions | evidence for the convergence and stability analysis |

## 4. Results

The Kaggle score uses the instructor's protocol: first 300 sorted images, Inception-v3 features, FID and the
script's "MiFID" (mean paired cosine distance), averaged over both directions. All other metrics come from
`gan_metrics.py`; their definitions are in that file's docstring.

<!-- METRICS:START -->
_Filled in automatically by `evaluate_local.py` after the full run._
<!-- METRICS:END -->

![loss curves](outputs/RUN_ID/loss_curves.png)
![epoch curves](outputs/RUN_ID/epoch_curves.png)
![photo to Monet](outputs/RUN_ID/final_samples_B2A.png)
![Monet to photo](outputs/RUN_ID/final_samples_A2B.png)

## 5. Training behaviour, convergence and stability (3.1.5, 3.2.3)

_Written after the full run, from `loss_curves.png`, `epoch_curves.png`, `steps.csv` and `epochs.csv`. It
covers:_
* the generator/discriminator balance;
* the cycle and identity losses over time;
* the FID trajectory;
* gradient norms and NaN steps;
* the effect of the LR decay.

## 6. Cycle-consistency verification (3.2.2)

* **Code level.** `tests/test_task3.py` checks that the cycle and identity terms are wired correctly
  (perfect reconstruction gives exactly 0), and that optimising the cycle loss alone drives the
  reconstruction error down.
* **Model level.** For the trained model, `evaluate_local.py` reports the cycle-reconstruction L1 and the
  LPIPS between each input and its reconstruction G_back(G(x)), in both directions. The third row of
  `final_samples_*.png` shows the reconstructions next to the inputs.

_Interpretation written after the full run._

## 7. Visual quality, human audit and discussion (3.2.1, 3.2.4, 3.2.6)

_Written after the full run and after the human audit (2 raters, 30 fixed blinded samples)._

## 8. Kaggle submission (3.2.5)

`submission.csv` is produced by `evaluate_local.py` from the direct outputs of my generators. There is no
editing, selection or external model, and the scored images are the deterministic translations of the first
300 sorted images of each domain. After uploading it to the class competition, the team name, public and
private score and rank are recorded in `kaggle_leaderboard.json`.

## 9. Comparison with teammates (team)

| Member | Generator | Discriminator | Losses (λ_cyc / λ_id) | Training | FID (avg) | KID B2A | Human audit |
|---|---|---|---|---|---|---|---|
| Shibin Thomas | ResNet-9, resize-conv upsampling | 70×70 PatchGAN | LSGAN + cycle 10 + identity 5 | 40 epochs × 2,000 pairs, batch 4 | | | |
| _teammate_ | | | | | | | |

## References
* Zhu, Park, Isola & Efros, *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017.
* Johnson, Alahi & Fei-Fei, *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*, ECCV 2016.
* Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (PatchGAN), CVPR 2017.
* Mao et al., *Least Squares Generative Adversarial Networks*, ICCV 2017.
* Shrivastava et al., *Learning from Simulated and Unsupervised Images through Adversarial Training* (image pool), CVPR 2017.
* Odena, Dumoulin & Olah, *Deconvolution and Checkerboard Artifacts*, Distill 2016.
* Ulyanov, Vedaldi & Lempitsky, *Instance Normalization: The Missing Ingredient for Fast Stylization*, 2016.
* Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID), NeurIPS 2017.
* Bińkowski et al., *Demystifying MMD GANs* (KID), ICLR 2018.
* Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative Models*, NeurIPS 2019.
* Naeem et al., *Reliable Fidelity and Diversity Metrics for Generative Models* (density / coverage), ICML 2020.
* Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS), CVPR 2018.
