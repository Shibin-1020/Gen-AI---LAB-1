"""Task 1.4 helper -- find candidate failure cases in the generated samples.

This does NOT write the failure analysis for you; it ranks the samples so you can pick
three real failures and describe them in failure_analysis.md. Heuristics:
  * repetition         -- high repeated-4-gram rate / most repeated phrases
  * broken words       -- word tokens never seen in the training split (misspellings, fused words)
  * loss of coherence  -- many distinct character names, or the continuation hit the length
                          limit without ending the story (<eos>)
  * broken grammar     -- very long sentences, doubled words ("the the"), unbalanced quotes
Outputs: outputs/<run_id>/failure_candidates.md  (ranked candidates per category)
         failure_analysis.md (member folder) -- a factual DRAFT with three distinct cases, written
         only while the file still contains the AUTO-DRAFT marker, i.e. never over your own edits.

Run:  python task1_llm/shibin_thomas/src/analyze_failures.py --config ... --run-id <run_id|latest>
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import numpy as np

from data_prep import decode, load_vocab
from shared_eval.metrics import most_repeated_ngrams, repeated_ngram_rate, words
from utils import MEMBER_DIR, latest_run_id, load_config, read_json, rel, repo_path, run_dirs

NAME_RE = re.compile(r"(?<![.!?\"]\s)(?<!^)\b([A-Z][a-z]+)\b")
SENT_RE = re.compile(r"[^.!?]+[.!?]")
COMMON_CAPS = {"The", "A", "An", "He", "She", "They", "It", "I", "We", "You", "One", "Once", "Then", "But",
               "And", "So", "When", "His", "Her", "There", "This", "That", "Mom", "Dad", "Mommy", "Daddy",
               "Yes", "No", "Oh", "Let", "What", "Why", "How", "After", "In", "At", "Every", "Finally"}


def training_lexicon(processed, idx_to_char, max_chars=20_000_000) -> Counter:
    meta = read_json(processed / "meta.json")
    arr = np.fromfile(processed / "train.bin", dtype=np.uint8 if meta["dtype"] == "uint8" else np.uint16)
    return Counter(words(decode(arr[:max_chars], idx_to_char)))


def analyse(sample: dict, lexicon: Counter) -> dict:
    text = sample["continuation"]
    toks = words(text)
    oov = sorted({w for w in toks if lexicon[w] == 0})
    names = {n for n in NAME_RE.findall(text) if n not in COMMON_CAPS}
    sents = SENT_RE.findall(text)
    long_sents = [s.strip() for s in sents if len(s.split()) > 35]
    doubled = re.findall(r"\b(\w+) \1\b", text.lower())
    return {
        "rep4": repeated_ngram_rate(text, 4) or 0.0,
        "top_repeats": most_repeated_ngrams(text, 4, 3),
        "oov_words": oov,
        "names": sorted(names),
        "no_eos": not sample["ended_with_eos"],
        "long_sentences": long_sents,
        "doubled_words": doubled,
        "unbalanced_quotes": text.count('"') % 2 == 1,
        "stutters": re.findall(r"([a-z]{2,})\1", text.lower()),    # e.g. "litlittle", "therere"
    }


AUTO_MARKER = "<!-- AUTO-DRAFT"


def _tag(s: dict) -> str:
    return s["mode"] + ("" if s["sample_idx"] is None else f" #{s['sample_idx']}")


def _snippet(s: dict) -> str:
    return "```text\n" + s["prompt"] + s["continuation"] + ("\n[EOS]" if s["ended_with_eos"] else "") + "\n```"


def draft_failure_analysis(rows, run_id, block_size, max_new) -> str:
    """Pick three distinct samples (repetition, broken words / grammar, coherence) and describe them factually."""
    mean_rep = sum(r["rep4"] for _, r in rows) / max(1, len(rows))
    used: set[int] = set()

    def pick(key):
        for i, (s, r) in sorted(enumerate(rows), key=lambda x: key(x[1][1])):
            if i not in used:
                used.add(i)
                return rows[i]
        return None

    cases = []
    s, r = pick(lambda r: -(10 * r["rep4"] + len(r["doubled_words"]) + 0.5 * len(r["stutters"])))
    if r["rep4"] > 0:
        obs = (f"Repeated-4-gram rate of this continuation is {r['rep4']:.2f} versus {mean_rep:.2f} on average over all "
               f"samples; most repeated phrases: {r['top_repeats']}. ")
    else:
        obs = (f"No whole phrase repeats (repeated-4-gram rate 0.00), but the text repeats at the word/character level: "
               f"doubled words {r['doubled_words'][:5] or 'none'}, stuttered character groups "
               f"{r['stutters'][:6] or 'none'} (a syllable emitted twice, as in 'litlittle'). ")
    if r["rep4"] == 0:
        obs += ("A character-level model has no notion of a finished word or syllable: after emitting 'lit' the most "
                "likely next characters are again the start of 'little', so the model can restart a word it has "
                "already begun. The effect is strongest early in training and at higher temperature.")
    elif s["mode"] == "greedy":
        obs += ("This is a greedy-decoded sample: argmax decoding always follows the single most likely character, so "
                "once the model produces a high-probability phrase it is drawn back into the same loop and cannot "
                "escape. Temperature sampling reduces this (compare the greedy vs sampled repeated-4-gram rate in "
                "metrics_report.csv).")
    else:
        obs += ("Even with temperature sampling the model falls back to a frequent TinyStories template phrase; it has "
                f"learned that such phrases are highly probable in general, and with only a {block_size}-character "
                "window it does not 'remember' having already used it a few sentences earlier.")
    cases.append(("Repetition", s, obs))

    s, r = pick(lambda r: -(len(r["oov_words"]) * 3 + len(r["doubled_words"]) * 2 + len(r["long_sentences"])
                            + int(r["unbalanced_quotes"])))
    if r["oov_words"]:
        kind = "Broken grammar / invented (misspelled) words"
        obs = (f"Words that never occur in my training split: {r['oov_words'][:10]}. A character-level model spells every "
               "word letter by letter; when a low-probability character is chosen mid-word, the model continues the "
               "corrupted prefix and produces a non-word, and the surrounding sentence often loses its grammatical "
               "structure.")
    else:
        kind = "Broken grammar"
        obs = (f"Doubled words: {r['doubled_words'][:5] or 'none'}; over-long sentences: {len(r['long_sentences'])}; "
               f"unbalanced quotes: {r['unbalanced_quotes']}. Local character patterns are fluent, but the model does "
               "not track sentence-level structure (subject/verb agreement, closing a quotation).")
    if r["doubled_words"] and r["oov_words"]:
        obs += f" It also contains doubled words: {r['doubled_words'][:5]}."
    cases.append((kind, s, obs))

    s, r = pick(lambda r: (-len(r["names"]), -int(r["no_eos"])))
    obs = f"Character names appearing in the continuation: {r['names'] or 'none'}. "
    if r["no_eos"]:
        obs += f"The story did not finish (no <eos>) within the {max_new}-character generation limit. "
    obs += (f"Generated text longer than the {block_size}-character context window means that earlier facts (who the "
            "characters are, what the problem was) literally fall out of the model's context, so it can introduce "
            "new characters, switch names or start a new plot instead of resolving the original one -- loss of "
            "coherence / entity hallucination rather than a spelling problem.")
    cases.append(("Loss of coherence / hallucinated entities", s, obs))

    out = [f"{AUTO_MARKER}: generated by src/analyze_failures.py from run {run_id}. Review and edit the observations "
           "in your own words; once you delete this line the file is never overwritten again. -->",
           "# Task 1.4 - Sequence model failure analysis", "",
           f"Run: `{run_id}` - samples: `outputs/{run_id}/samples.md` - ranked candidates: "
           f"`outputs/{run_id}/failure_candidates.md`.", "",
           "Generated with the shared prompts (`task1_llm/shared_eval/prompts.json`), greedy decoding and temperature "
           "0.8 sampling from the best checkpoint. Prompt text is shown at the start of each snippet.", ""]
    for n, (kind, s, obs) in enumerate(cases, 1):
        out += [f"## Failure case {n}: {kind}", "",
                f"**Prompt:** `{s['prompt']}` - **decoding:** {_tag(s)}", "", _snippet(s), "",
                f"**Failure type:** {kind}", "", f"**Observation:** {obs}", ""]
    out += ["## Summary and testable fixes", "",
            "| Failure | Likely cause | Testable fix |", "|---|---|---|",
            "| Repetition | argmax / low-entropy decoding, short context | temperature or top-k / nucleus sampling, "
            "repetition penalty; compare repeated-4-gram rate |",
            "| Invented words / grammar | character-level spelling, one bad sample corrupts the word | lower temperature or "
            "top-k; larger model / more epochs; sub-word tokenizer |",
            "| Loss of coherence | 256-char context window, small model | longer block_size, more layers; measure with "
            "longer-range eval |", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--run-id", default="latest")
    p.add_argument("--top", type=int, default=4)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    run_id = latest_run_id(cfg["run_name"]) if a.run_id == "latest" else a.run_id
    out = run_dirs(run_id)["outputs"]
    samples = read_json(out / "samples.json")
    processed = repo_path(cfg["data"]["processed_dir"])
    _, idx_to_char = load_vocab(processed)
    lex = training_lexicon(processed, idx_to_char)
    rows = [(s, analyse(s, lex)) for s in samples]

    def block(s, r):
        return (f"- **{s['prompt']!r} / {s['mode']}{'' if s['sample_idx'] is None else ' #' + str(s['sample_idx'])}**"
                f" - rep4={r['rep4']:.2f}, repeats={r['top_repeats']}, OOV={r['oov_words'][:8]}, "
                f"names={r['names']}, no_eos={r['no_eos']}, doubled={r['doubled_words'][:5]}, "
                f"unbalanced_quotes={r['unbalanced_quotes']}\n\n  ```text\n  {s['prompt']}"
                + s["continuation"].replace("\n", "\n  ") + "\n  ```\n")

    cats = {
        "Repetition (highest repeated-4-gram rate)": sorted(rows, key=lambda x: -x[1]["rep4"]),
        "Broken / invented words (most out-of-vocabulary words)": sorted(rows, key=lambda x: -len(x[1]["oov_words"])),
        "Loss of coherence (most character names; no <eos>)": sorted(
            rows, key=lambda x: (-len(x[1]["names"]), -int(x[1]["no_eos"]))),
        "Broken grammar (long sentences, doubled words, unbalanced quotes)": sorted(
            rows, key=lambda x: -(len(x[1]["long_sentences"]) + len(x[1]["doubled_words"])
                                  + int(x[1]["unbalanced_quotes"]))),
    }
    with open(out / "failure_candidates.md", "w", encoding="utf-8") as f:
        f.write(f"# Failure candidates - {run_id}\n\nAutomatically ranked from `samples.json`. "
                "Read them, pick three genuine failures, and write them up in `failure_analysis.md`.\n\n")
        for title, ranked in cats.items():
            f.write(f"## {title}\n\n")
            for s, r in ranked[:a.top]:
                f.write(block(s, r) + "\n")
    print(f"wrote {rel(out / 'failure_candidates.md')}")

    fa = MEMBER_DIR / "failure_analysis.md"
    if cfg.get("smoke", False):
        return
    if fa.exists() and AUTO_MARKER not in fa.read_text(encoding="utf-8"):
        print(f"{rel(fa)} has been edited by hand - not overwritten")
        return
    fa.write_text(draft_failure_analysis(rows, run_id, cfg["model"]["block_size"],
                                         cfg["generation"]["max_new_tokens"]), encoding="utf-8")
    print(f"wrote draft {rel(fa)} - review it and put the observations in your own words")


if __name__ == "__main__":
    main()
