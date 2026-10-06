# Task 3 — CycleGAN image style transfer (Monet ↔ Photo, Kaggle class competition)

```
task3_gan/
├── data/                           shared dataset from the class zip (git-ignored images)
│   ├── monet_jpg/                  300 real Monet paintings   = domain A
│   ├── photo_jpg/                  7,038 real photos          = domain B
│   └── Part3_Evaluation_Script.ipynb   instructor's FID / MiFID script (defines the Kaggle metric)
└── <member_name>/                  each member's own CycleGAN (see shibin_thomas/)
```

**Getting the data.** Unzip the class `Part3_export.zip`, then the `dataset.zip` inside it. Move
`dataset/monet_jpg` and `dataset/photo_jpg` to `task3_gan/data/`, and delete `__MACOSX/` and `.DS_Store`
(macOS metadata, not images).

## Team evaluation protocol
* **Kaggle metric:** the instructor's script.
  * FID and "MiFID" (mean paired cosine distance of Inception-v3 features) for `pred_B2A` vs real Monet and
    for `pred_A2B` vs real photos.
  * First 300 sorted images of each folder; the two directions are averaged into `submission.csv`.
* **Each member's `evaluate_local.py`** re-implements that protocol exactly and adds:
  * KID, generative precision/recall, density/coverage;
  * cycle-reconstruction L1, LPIPS, content cosine;
  * training statistics.
* **Integrity:** submitted images are the direct outputs of each member's own CycleGAN. No editing,
  selection or pretrained generative models; pretrained networks are used only to measure.
* **Human audit:** 30 fixed blinded samples per member, rated by 2 raters on style / content / artifacts
  (1–5), with quadratic-weighted Cohen's κ.

## Members
| Member | Folder | Generator | Discriminator | Kaggle score | Rank |
|---|---|---|---|---|---|
| Shibin Thomas | [`shibin_thomas/`](shibin_thomas/) | ResNet-9, resize-conv upsampling | 70×70 PatchGAN | recorded in `kaggle_leaderboard.json` | recorded in report |
| Denisha Ketan Tank | [`member_denisha/`](member_denisha/) | ResNet-9 CycleGAN | PatchGAN | -50.6033 | 34 |

Denisha's official evaluator outputs are in `member_denisha/outputs/submission_metrics_official.json`,
and the expanded metric table is `member_denisha/outputs/full_metrics_report.csv`. The exported
archive contains 300 competition images. The 2-rater human audit is intentionally marked incomplete
until Rater 2 independently fills the second score columns and the agreement script produces its JSON.
