# Demo cheat sheet (Shibin Thomas, Team 09)

## Task 1: character-level GPT (TinyStories)

**Live demo**
* `python3 demo/demo_task1_generate.py`
* `--greedy` shows the repetition failure.

**Architecture**
* Decoder-only Transformer, 6 layers, width 320, 8 heads (head dim 40), FFN 1,280 with GELU.
* Context of 256 characters, pre-LayerNorm, tied embeddings.
* 7.51M parameters.
* Attention, LayerNorm, FFN and residual blocks are written by hand (`src/model.py`).
* The causal mask is a lower-triangular matrix: future positions are set to −inf before the softmax.
* `tests/test_model.py` checks that no PyTorch attention or Transformer module is used.

**Why these choices**
* 256 characters is 2–3 sentences, and attention cost grows with T².
* ~7.5M parameters for 0.9B training characters is ~120 characters per parameter, so there is little over-fitting risk, and training takes 29 minutes.
* Pre-LayerNorm is stable with a short warm-up.

**Training**
* AdamW (β 0.9, 0.95), weight decay 0.1, peak LR 6e-4.
* 2% warm-up, then cosine decay to 6e-5.
* Batch 64 × 256, gradient clip 1.0, bf16, 10 epochs on an RTX 5090.

**Data**
* 2.12M stories, Unicode mapped to ASCII, non-ASCII or very short stories dropped: 1.99M kept.
* Own split of 100K train / 10K validation (seed 266).
* Vocabulary of 90 characters (including `<eos>` and `<unk>`), built from the training split only.

**Metrics: how each is computed**

| Metric | How it is computed | Value |
|---|---|---|
| Cross-entropy (CE) | mean −log p(next character), in nats | val 0.571, train 0.559 |
| Perplexity | exp(CE): on average ~1.77 plausible next characters | 1.77 |
| Bits per character (BPC) | CE / ln 2 | 0.824 |
| Generalisation gap | val CE − train CE, both measured without dropout | 0.012 (no over-fitting) |
| Top-1 accuracy | the argmax character equals the true next character | 81.7% |
| Distinct-n | unique word n-grams / all word n-grams in the samples | 0.23 / 0.70 / 0.93 |
| Repeated 4-gram rate | 1 − unique / total word 4-grams in a sample | 0.007 sampled, 0.137 greedy |
| Gradient norm | global L2 norm before clipping | mean 0.30, max 14.7 at step 0; all clipped steps in warm-up; 0 NaN; 0 loss spikes |
| Speed | throughput | 570K training characters/s; 255 characters/s generation (batch 1, no KV cache) |
| Memory and time | peak GPU memory, total training time | 3.3 GB; 28.9 min |

**Failure cases**
1. **Greedy repetition:** "the tree was very high…" repeated; argmax decoding locks into a loop.
2. **Rare-word spelling:** lute / lutter / luter / lutte; the model spells each word letter by letter.
3. **Coherence loss beyond 256 characters:** the story drifts from the zoo to a diamond, and new characters appear.

**Differences from Denisha**
* Denisha: 4 layers, width 256, 4 heads, context 128, 3.25M parameters, AdamW (0.9, 0.999), LR 5e-4, 500-step warm-up, batch 48, fp16, T4.
* Her model is smaller and has half the context, giving BPC 0.984 against 0.824.

## Task 2: Yelp sentiment (no pretrained embeddings)

**Live demo**
* `python3 demo/demo_task2_classify.py`
* The "not bad at all" example: the baseline says negative, the BiGRU says positive. Word order matters.

**Preprocessing**
* Remove duplicates and reviews that also occur in the test set.
* Lowercase, expand contractions (didn't → did not), remove punctuation.
* Remove stopwords but **keep negations and contrast words**, then apply Snowball stemming.
* Vocabulary of 30K tokens built from the training split only.
* 256 tokens per review, using the first 128 + last 128 tokens.
* 128-dimensional embeddings learned from scratch.

**Models**

| Model | Architecture | Why |
|---|---|---|
| Baseline | mean of embeddings → MLP 64 | a bag of words; fast; best calibrated |
| TextCNN | filters of width 3/4/5 × 128, max-over-time pooling | detects local phrases such as "not worth" |
| BiGRU + attention | 2-layer bidirectional GRU (128 per direction), additive attention | reads the whole review in order; attention weights the verdict |

**Results on the official test set (38K reviews)**
* Accuracy: 93.04 / 94.67 / **95.58%**.
* MCC: 0.861 / 0.893 / 0.912.
* ROC-AUC: 0.980 / 0.988 / 0.992.

**What the metrics mean**
* **Macro/micro/weighted:** all equal here because the test set is exactly balanced.
* **MCC:** correlation between prediction and truth, from −1 to 1.
* **Brier score:** mean squared error of the predicted probability.
* **ECE:** gap between confidence and accuracy, over 15 bins.
* **Bootstrap CI:** 1,000 resamples of the test set.
* **McNemar test:** compares only the reviews where exactly one of two models is right. BiGRU vs baseline: p = 4.7e-99.

**Where the gains come from**
* Compared with the baseline, the BiGRU removes 40% of errors on reviews with negation and 38% on reviews with a contrast word.

**Failure cases (20 reviewed)**
* Mixed sentiment (7), label noise (5: the text contradicts the stars), faint praise (3).
* One each of sarcasm ("Keepin it real dumpy!"), the idiom "to die for", and complex negation.

**Differences from Denisha**
* Denisha's baseline uses MLP 128; her GRU has no attention and reads the last token.
* Batch 512, reduce-on-plateau schedule, 12 epochs, bf16.
* Her TextCNN has the same layer sizes but different hyperparameters.
* She evaluated on her own 112K hold-out split, not the official test set.
* Her best model: GRU at 95.19%.

## Task 3: CycleGAN, Monet ↔ photo (both members speak)

**Live demo**
* `python3 demo/demo_task3_translate.py`
* The grid shows input | translation | reconstruction.

**Architecture**
* Generators: ResNet-9 (11.38M parameters each), with resize-convolution instead of transposed convolutions to avoid checkerboard artifacts. InstanceNorm and reflection padding.
* Discriminators: 70×70 PatchGAN (2.76M each), judging local texture.

**Losses**
* LSGAN adversarial loss.
* Cycle loss, weight 10: G_BA(G_AB(a)) ≈ a.
* Identity loss, weight 5: G_BA(a) ≈ a.
* The discriminators use a 50-image pool of earlier fakes.

**Training**
* Adam 2e-4 (0.5, 0.999).
* 20 epochs at constant LR, then 20 with linear decay.
* 2,000 unpaired pairs per epoch, batch 4, fp32, 91 minutes.

**Metrics: how each is computed**
* **FID:** Inception-v3 pool features (2048-d), first 300 sorted images. FID = ‖μr−μg‖² + Tr(Σr+Σg−2(ΣrΣg)^½).
* **MiFID:** in the instructor script, the mean cosine distance between paired real and generated features.
* **Kaggle score:** (FID + MiFID) / 2.
* **Values:** FID 118.07 (photo→Monet) and 117.34 (Monet→photo), average **117.71**; MiFID 0.42; score **59.06**. The team's best entry is Denisha's: −50.60, rank 34.
* **KID:** unbiased MMD with a cubic kernel; works with small samples.
* **Precision and recall:** k-NN manifolds. Photo→Monet precision is 0.27 (low realism); Monet→photo recall is 0.32 (low diversity).
* **Cycle L1:** 0.06. **LPIPS:** perceptual distance. **Content cosine:** 0.75–0.79.
* **Human audit:** you rated as rater 1 for your model, and rater 2 for Denisha's.

**Training behaviour**
* The discriminators drift below the 0.25 balance point; D_A (Monet) reaches 0.089 because it memorises the 300 paintings.
* FID flattens at about 120 after epoch 25.

**Failure cases**
1. Dark photos are washed out.
2. Reds and oranges turn blue, yet the reconstruction recovers them: "steganography".
3. A stipple texture appears instead of brush strokes.
4. Foggy paintings become saturated sunsets.
5. Monet→photo outputs still look painted.

**Differences from Denisha (why her FID is better, 100.8)**
* 60 epochs instead of 40.
* Identity weight 2.5 instead of 5, so the colours can shift more.
* Transposed-convolution upsampling.
* fp16 on an RTX 4090.

Her outputs are smoother, with colours closer to the input.

**Next step:** the prepared v2 config adds DiffAugment, generator EMA, identity weight 2.5, 60 epochs and held-out epoch selection.
