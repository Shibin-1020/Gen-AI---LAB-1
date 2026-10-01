# Task 1.4 — Sequence model failure analysis · Shibin Thomas

**Model:** best checkpoint of run `gpt_char_v1_20261001-153007` (epoch 10, val CE 0.571 nats/char).
**Generation:** the 10 shared team prompts (`task1_llm/shared_eval/prompts.json`), each with one greedy
continuation and three temperature-0.8 samples, 500 new characters each (40 samples in total).
All samples: [`outputs/gpt_char_v1_20261001-153007/samples.md`](outputs/gpt_char_v1_20261001-153007/samples.md).
Ranked candidates (repetition, unusual words, name changes, grammar signals):
[`outputs/gpt_char_v1_20261001-153007/failure_candidates.md`](outputs/gpt_char_v1_20261001-153007/failure_candidates.md).

Overall, most samples are fluent, grammatical TinyStories-style text. The three cases below are the
clearest failures I found. Each one shows a different weakness of a small character-level model.

---

## Failure case 1 — Repetition (degenerate loop)

**Prompt:** `In a small house near the forest,` · **decoding:** greedy

```text
In a small house near the forest, there was a big tree. The tree was very high and the tree was very high.
The tree was very high up in the tree. The tree was very high and the tree was very high.

One day, the tree saw a big tree. The tree was very high and the tree was very high. The tree was very high
and the tree was very high. The tree was very high and the tree was very high. The tree was very high and
the tree was very high.

The tree was very happy to see the tree and the tree was very happy. The tree was very proud of
```

**Failure type:** Repetition.

**Observation:**
* The phrase "the tree was very" occurs 16 times and "tree was very high" 13 times. The repeated 4-gram rate
  of this continuation is 0.61, against an average of 0.04 over all 40 samples.
* The story also gets stuck on a single entity: "the tree saw a big tree" and "happy to see the tree".
* This comes from greedy decoding. At every step the model takes the single most likely next character.
  "The tree was very high" is a high-probability phrase, so once the model has written it, the most likely
  continuation is to start it again. Greedy decoding has no randomness that could break the loop.
* The same model with temperature-0.8 sampling barely repeats itself. Over the whole run, the repeated
  4-gram rate is 0.137 for greedy decoding and 0.007 for sampling (`metrics_report.csv`).
* **Testable fix:** sample instead of taking the argmax (temperature or top-k), or add a repetition penalty.
  Then compare the repeated 4-gram rate on the same prompts.

## Failure case 2 — Broken words / inconsistent spelling

**Prompt:** `In a small house near the forest,` · **decoding:** temperature 0.8 (sample #2)

```text
In a small house near the forest, there was a big field. The thief had fallen over the cart.

"Can you help me with my tooth?" asked Tim.

Tim was scared. He ran and ran until he came across a big lute. He saw a small tree which was very pretty.
Tim wanted to take the lutter from him. He said yes, but he had to be careful.

The luter was big and boring. Tim saw that the lutte was scary and hurt. He wanted to help his friends so
he could move to the lutter which was his favorite thing. After that, Tim learned that being a lute a
```

**Failure type:** Broken grammar / invented words (misspelling).

**Observation:**
* The model introduces an object, "a big lute", and then spells it four different ways within a few
  sentences: **lute → lutter → luter → lutte**. "Lutter", "luter" and "lutte" are not English words.
  "Lute" is a real word but very rare in children's stories.
* A character-level model has no fixed list of words; it spells every word one letter at a time. For a
  rare word it has weak statistics, so each time the word comes back the model rebuilds it from
  frequent letter patterns ("-er", "-tte") and lands on a different spelling.
* The grammar breaks at the same time: "He said yes" has no question to answer, "the lutte was scary and
  hurt" makes no sense, and the moral "being a lute a…" is meaningless.
* The opening is also incoherent ("Can you help me with my tooth?"). Sampling at temperature 0.8 lets
  less likely characters through, and here that cost meaning.
* **Testable fix:** lower the temperature or use top-k sampling for fewer rare-character errors; train
  longer or use a bigger model (validation loss was still falling at epoch 10); or use a sub-word tokenizer
  so that whole words are single tokens.

## Failure case 3 — Loss of coherence / entity drift

**Prompt:** `After school, Sam wanted to` · **decoding:** temperature 0.8 (sample #1)

```text
After school, Sam wanted to go to the zoo and see all the animals.

They all walked closer and held on tight. Sam thought it was tremommeter than the diamond. They all had fun
playing hide and seek, and Sam wanted to eat the diamond he really wanted.

Together, Jack and Sam started to fill their bucket with food. He was so proud of their work!

Soon, their mother came and they were all so happy. The town was perfect, and they had the best place ever.

Tom and Sam continued to throw their bucket until it was time to go hom
```

**Failure type:** Loss of coherence (and hallucinated entities).

**Observation:**
* Every sentence is locally fluent, but the story has no consistent plot:
  * It starts at the zoo, then switches without explanation to a diamond, hide and seek, a bucket of food
    and "the town".
  * It invents companions that were never introduced: "They all", then **Jack**, then **Tom**, playing
    alongside Sam.
  * Pronouns stop matching: "He was so proud of their work".
* Meaning breaks down ("Sam wanted to eat the diamond"), and it contains an invented word, "tremommeter"
  (it looks like a merge of "tremendous" and "thermometer").
* The story never ends: no `<eos>` within the 500-character limit, cut off at "go hom".
* The main cause is the context window. The model sees only the last **256 characters**, about 2–3
  sentences, and this continuation is 500 characters long. By the second paragraph, the zoo and the
  original setting have literally fallen out of the model's view, so it cannot stay consistent with
  them. With 7.5M parameters it also has limited capacity for long-range plot structure.
* **Testable fix:** a longer `block_size` (e.g. 512) or more layers. Measure coherence on long
  generations, for example the number of distinct character names per story.

---

## Summary

| # | Failure type | Where it appears | Root cause | Testable fix |
|---|---|---|---|---|
| 1 | Repetition | greedy decoding (repeated 4-gram rate 0.137 vs 0.007 for sampling) | argmax decoding locks into high-probability loops | temperature / top-k sampling, repetition penalty |
| 2 | Broken words / grammar | temperature sampling, rare words | letter-by-letter spelling; weak statistics for rare words | lower temperature / top-k, longer training, sub-word tokens |
| 3 | Loss of coherence | long continuations (> 256 chars) | finite 256-char context window, small model | longer context, deeper model |
