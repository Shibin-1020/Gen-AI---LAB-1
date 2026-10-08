"""Task 1 live demo: load Shibin's character-level GPT checkpoint and generate text from a prompt.

    python demo/demo_task1_generate.py
    python demo/demo_task1_generate.py --prompt "Once upon a time" --chars 300 --temperature 0.8
    python demo/demo_task1_generate.py --greedy

Runs on CPU in a few seconds; no dataset needed (vocabulary and checkpoint are in the repo).
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MEMBER = ROOT / "task1_llm/shibin_thomas"
sys.path.insert(0, str(MEMBER / "src"))
from data_prep import EOS_ID, decode, encode, load_vocab  # noqa: E402
from model import GPTCharLM, GPTConfig  # noqa: E402

CKPT = MEMBER / "checkpoints/gpt_char_v1_20261001-153007/best_model.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--chars", type=int, default=300, help="new characters to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=266)
    a = p.parse_args()

    t0 = time.time()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GPTCharLM(GPTConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    c2i, i2c = load_vocab(MEMBER / "data_processed/main")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {CKPT.relative_to(ROOT)} (epoch {ck.get('epoch', '?')}, {n_params:,} parameters) "
          f"in {time.time() - t0:.1f}s")
    print(f"Config: {ck['model_config']}\n")

    gen = torch.Generator().manual_seed(a.seed)
    ids = torch.tensor([encode(a.prompt, c2i)])
    for k in range(1 if a.greedy else a.samples):
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(ids, a.chars, temperature=a.temperature, greedy=a.greedy, eos_id=EOS_ID,
                                 generator=gen)
        text = decode(out[0, ids.size(1):].tolist(), i2c, stop_at_eos=True)
        mode = "greedy" if a.greedy else f"temperature {a.temperature}"
        print(f"--- sample {k + 1} ({mode}, {time.time() - t1:.1f}s) ---")
        print(a.prompt + text, "\n")


if __name__ == "__main__":
    main()
