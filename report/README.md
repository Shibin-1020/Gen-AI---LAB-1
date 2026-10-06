# Final report

`DATA266_Lab1_Report_Team_09.pdf` is the combined team report (brief, Section 6).

It is built from Denisha's version of the report, kept unchanged, plus Shibin Thomas's matching
supplementary pages:
```bash
pip install reportlab pypdf pillow
python report/build_final_report.py
```
* `sources/DATA266_Lab1_Report_Team_09_denisha_version.pdf`: Denisha's version. All 30 pages are copied unchanged.
* `build_final_report.py` inserts Shibin's pages next to Denisha's matching pages:
  * contribution overview;
  * Task 1, 2 and 3 detailed findings;
  * representative images;
  * human audit;
  * final contribution checklist.

  The numbers come from Shibin's committed result files.
* `human_audit/`: the completed two-rater audit of Denisha's model and its agreement values.
* `build_report.py` / `team_info.yaml`: an earlier integrated version of the report. Its output goes to `drafts/`.
