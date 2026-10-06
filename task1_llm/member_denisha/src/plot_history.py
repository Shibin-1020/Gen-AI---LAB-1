import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--history', default='outputs/history.json'); ap.add_argument('--out', default='outputs/loss_curves.png'); args = ap.parse_args()
h = json.loads(Path(args.history).read_text()); epochs = [r['epoch'] for r in h]
plt.plot(epochs, [r['train_loss'] for r in h], label='train'); plt.plot(epochs, [r['val_loss'] for r in h], label='validation'); plt.xlabel('Epoch'); plt.ylabel('Cross-entropy loss'); plt.legend(); plt.tight_layout(); plt.savefig(args.out, dpi=160)
