"""Compute agreement for the completed two-rater human audit CSV."""
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ap = argparse.ArgumentParser(); ap.add_argument('--csv', default='outputs/human_audit.csv'); ap.add_argument('--out', default='outputs/human_audit_metrics.json'); args = ap.parse_args()
df = pd.read_csv(args.csv)
required = ['style_score_1','style_score_2','content_score_1','content_score_2','artifact_score_1','artifact_score_2']
missing = [c for c in required if c not in df or df[c].isna().all() or (df[c].astype(str).str.strip() == '').all()]
if missing: raise ValueError(f'Complete both raters before computing agreement: {missing}')
result = {}
for dimension in ['style','content','artifact']:
    a, b = df[f'{dimension}_score_1'], df[f'{dimension}_score_2']; result[dimension] = {'cohen_kappa': float(cohen_kappa_score(a, b)), 'percent_agreement': float((a == b).mean())}
result['n_samples'] = int(len(df)); Path(args.out).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
