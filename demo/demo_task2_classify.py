"""Task 2 live demo: classify reviews with Shibin's three sentiment models (no pretrained embeddings).

    python demo/demo_task2_classify.py
    python demo/demo_task2_classify.py --text "The food was not good at all, but the staff were lovely."

Uses the same preprocessing as training (configs/data.yaml) and the vocabulary built from the training split.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MEMBER = ROOT / "task2_sentiment/shibin_thomas"
sys.path.insert(0, str(MEMBER / "src"))
from data_prep import UNK_ID, Preprocessor, encode, load_vocab, normalize_raw  # noqa: E402
from models import build_model  # noqa: E402

RUN = MEMBER / "checkpoints/yelp3_20261001-182849"
MODELS = ["baseline_meanpool", "exp1_textcnn", "exp2_bigru_attn"]
EXAMPLES = [
    "The pizza was amazing and the staff were super friendly. Will definitely come back!",
    "Worst service ever. We waited an hour and the food was cold.",
    "The food was not bad at all, I would recommend it.",
    "Great location, but the food was bland and overpriced. Not worth it.",
    "For being a DUMP, should expect much more. Flys, stink, garbage. Keepin it real dumpy!",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="append", help="review text (can be given several times)")
    a = p.parse_args()
    texts = a.text or EXAMPLES

    dcfg = yaml.safe_load(open(MEMBER / "configs/data.yaml", encoding="utf-8"))
    pre = Preprocessor(dcfg)
    itos = load_vocab(MEMBER / "data_processed/main")
    stoi = {w: i for i, w in enumerate(itos)}
    t0 = time.time()
    models = {}
    for name in MODELS:
        ck_path = next((RUN / name).glob("*.pt"))
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        m = build_model(ck["model_config"], ck["vocab_size"])
        m.load_state_dict(ck["model"])
        models[name] = m.eval()
    print(f"Loaded 3 checkpoints from {RUN.relative_to(ROOT)} in {time.time() - t0:.1f}s "
          f"(vocabulary {len(itos):,} tokens)\n")

    for text in texts:
        toks = pre.tokens(normalize_raw(text) or "")
        ids = encode(toks, stoi, dcfg["max_len"], dcfg.get("head_tokens", 128), dcfg.get("truncation", "head_tail"))
        x = torch.tensor([ids])
        lengths = torch.tensor([len(ids)])
        unk = sum(1 for i in ids if i == UNK_ID)
        print(f"Review: {text}")
        print(f"  tokens after preprocessing: {' '.join(toks)}  ({unk} unknown)")
        with torch.no_grad():
            for name, m in models.items():
                prob = torch.sigmoid(m(x, lengths)).item()
                print(f"  {name:18s} P(positive) = {prob:.3f} -> {'POSITIVE' if prob >= 0.5 else 'NEGATIVE'}")
        print()


if __name__ == "__main__":
    main()
