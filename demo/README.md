# Live demo (Shibin Thomas): runs offline on a laptop CPU

Each script loads a trained checkpoint from this repository and produces outputs in seconds.
No dataset or GPU is needed.

| Task | Command | What it shows |
|---|---|---|
| 1 | `python3 demo/demo_task1_generate.py --prompt "Once upon a time"` | char-level GPT (7.5M params) generating stories; add `--greedy` to show the repetition failure |
| 2 | `python3 demo/demo_task2_classify.py --text "The food was not bad at all"` | all three classifiers' P(positive) for any review; the default examples include negation and sarcasm |
| 3 | `python3 demo/demo_task3_translate.py` | CycleGAN photo->Monet and Monet->photo on 8 sample images; grids saved to `demo/outputs/` |

* **Task 3 with your own photo:** `python3 demo/demo_task3_translate.py --images path/to/photo.jpg`
* **Sample inputs:** the 8 images in `sample_inputs/` are cropped from the committed human-audit panels.
* **Timings on a CPU:**
  * checkpoint loading takes 3 s (Task 1) and 7 s (Task 3);
  * 300 characters take about 5 s;
  * 4 images take about 3 s.

## One-time setup on the demo laptop (macOS)
```bash
git clone -b shibin-lab1 https://github.com/Shibin-1020/Gen-AI---LAB-1.git
cd Gen-AI---LAB-1
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch torchvision pyyaml nltk scikit-learn pillow numpy
python3 demo/demo_task1_generate.py
python3 demo/demo_task2_classify.py
python3 demo/demo_task3_translate.py
```
Run all three once before the demo, so that everything is installed and cached locally.
