import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--history', default='outputs/history.json'); ap.add_argument('--out', default='outputs/loss_curves.png'); args = ap.parse_args()
h = json.loads(Path(args.history).read_text()); epochs = [r['epoch'] for r in h]
for key in ['g', 'd_a', 'd_b', 'cycle', 'identity']:
    plt.plot(epochs, [r[key] for r in h], label=key)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout(); plt.savefig(args.out, dpi=160)
