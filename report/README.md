# Final report

`DATA266_Lab1_Report_Team_09.pdf` is the combined team report (brief, Section 6).

It is generated, not hand-edited:
```bash
pip install reportlab pyyaml pillow
python report/build_report.py
```
* `build_report.py` reads every number for Shibin Thomas's models directly from the committed result files
  (`metrics_report.csv`, `comparison.csv`, `full_metrics_report.csv`, `epochs.csv`, manifests). The report
  therefore always matches the evidence, and each table names its source file.
* Teammates' rows, the team number and the repository link are taken from `team_info.yaml`; values left
  empty are shown as "pending".
* The Kaggle score and rank come from `task3_gan/shibin_thomas/kaggle_leaderboard.json`.
* Fonts: DejaVu Sans (bundled in `fonts/`, free licence in `fonts/LICENSE-DejaVu.txt`).

To update the report: edit `team_info.yaml` or the leaderboard JSON, run the command above, and commit the new PDF.
