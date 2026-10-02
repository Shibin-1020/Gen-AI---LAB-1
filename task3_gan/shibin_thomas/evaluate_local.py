"""Task 3.2 -- local evaluation of my CycleGAN (both directions) + Kaggle submission file.

    python task3_gan/shibin_thomas/evaluate_local.py --run-id <run_id|latest>

1. Kaggle metric, exactly as the instructor's Part3_Evaluation_Script.ipynb: FID and "MiFID" for
   Photo->Monet (real Monet vs pred_B2A) and Monet->Photo (real photos vs pred_A2B) on the first
   N_EVAL = 300 sorted images, averaged -> submission.csv (columns ID, FID, MiFID).
2. Everything else the brief asks for, per direction: KID, generative precision / recall, density /
   coverage, cycle-reconstruction L1, LPIPS (input vs translation, input vs reconstruction), content-
   preservation cosine similarity; training statistics from the raw logs (generator / discriminator /
   cycle / identity losses, gradient norms, NaN count, parameters, training time, images/sec, peak
   memory); human-audit scores + inter-rater agreement (once both raters are done); Kaggle leaderboard
   score + rank (from kaggle_leaderboard.json, filled in by hand after submitting).
3. Plots (loss curves, gradient norms, periodic FID, final samples), visual failure candidates, and
   full_metrics_report.csv (+ metrics_report.csv); results.md tables are refreshed for non-smoke runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import matplotlib  # noqa: E402
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import list_images, load_image, eval_transform  # noqa: E402
from gan_metrics import (InceptionFeatures, LPIPSMetric, instructor_fid_mifid, kid,  # noqa: E402
                         paired_cosine_similarity, precision_recall_density_coverage)
from train import save_grid  # noqa: E402
from translate import load_generator  # noqa: E402
from utils import (MEMBER_DIR, get_device, get_logger, latest_run_id, load_config, read_json, rel,  # noqa: E402
                   repo_path, run_dirs, set_seed, update_manifest, write_json)

AUTO_MARKER = "<!-- AUTO-DRAFT"
DIRECTIONS = {  # name: (input domain dir key, prediction folder, real target dir key, generator)
    "B2A (photo->monet)": ("photo_dir", "pred_B2A", "monet_dir", "G_BA", "G_AB"),
    "A2B (monet->photo)": ("monet_dir", "pred_A2B", "photo_dir", "G_AB", "G_BA"),
}


def read_steps(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    last = {int(r["step"]): r for r in rows}                    # after --resume keep the latest row per step
    rows = [last[k] for k in sorted(last)]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(raw_logs: Path, out: Path) -> None:
    s = read_steps(raw_logs / "steps.csv")
    w = max(1, len(s["step"]) // 100)
    x = s["step"][w - 1:]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for k, lab in (("gan_A2B", "G_AB adversarial (fool D_B)"), ("gan_B2A", "G_BA adversarial (fool D_A)"),
                   ("loss_D_A", "D_A (Monet)"), ("loss_D_B", "D_B (Photo)")):
        axes[0, 0].plot(x, smooth(s[k], w), label=lab)
    axes[0, 0].set(title="Adversarial losses (LSGAN; equilibrium ~0.25 for D, ~1 for G)", xlabel="step")
    for k, lab in (("cyc_A", "cycle A: |G_BA(G_AB(a)) - a|"), ("cyc_B", "cycle B: |G_AB(G_BA(b)) - b|"),
                   ("idt_A", "identity A: |G_BA(a) - a|"), ("idt_B", "identity B: |G_AB(b) - b|")):
        axes[0, 1].plot(x, smooth(s[k], w), label=lab)
    axes[0, 1].set(title="Cycle-consistency and identity L1 losses", xlabel="step")
    axes[1, 0].plot(x, smooth(s["loss_G"], w), color="#1f5aa6", label="total generator loss")
    ax2 = axes[1, 0].twinx()
    ax2.plot(s["step"], s["lr"], color="#7d3c98", lw=0.8, label="learning rate")
    axes[1, 0].set(title="Total generator loss and LR schedule", xlabel="step")
    ax2.set_ylabel("lr")
    axes[1, 1].plot(s["step"], s["grad_norm_G"], lw=0.4, alpha=0.6, label="||grad G||")
    axes[1, 1].plot(s["step"], s["grad_norm_D"], lw=0.4, alpha=0.6, label="||grad D||")
    axes[1, 1].set(title="Gradient norms (pre-update)", xlabel="step", yscale="log")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=140)
    plt.close(fig)

    ep = list(csv.DictReader(open(raw_logs / "epochs.csv", encoding="utf-8")))
    e = [int(r["epoch"]) for r in ep]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fe = [(int(r["epoch"]), float(r["fid_B2A"]), float(r["fid_A2B"])) for r in ep if r["fid_B2A"]]
    if fe:
        axes[0].plot([f[0] for f in fe], [f[1] for f in fe], "o-", label="FID photo->Monet (B2A)")
        axes[0].plot([f[0] for f in fe], [f[2] for f in fe], "s-", label="FID Monet->photo (A2B)")
    axes[0].set(title="Periodic FID during training (300 images)", xlabel="epoch", ylabel="FID")
    axes[1].plot(e, [float(r["cycle_l1_A"]) for r in ep], "o-", label="cycle L1 A (Monet)")
    axes[1].plot(e, [float(r["cycle_l1_B"]) for r in ep], "s-", label="cycle L1 B (photo)")
    axes[1].set(title="Cycle-reconstruction L1 on a fixed batch", xlabel="epoch", ylabel="mean |x - x_rec| (0-1)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "epoch_curves.png", dpi=140)
    plt.close(fig)


def training_stats(raw_logs: Path, summary: dict) -> dict:
    s = read_steps(raw_logs / "steps.csv")
    ep = list(csv.DictReader(open(raw_logs / "epochs.csv", encoding="utf-8")))[-1]
    out = {f"final_epoch_mean_{k}": float(ep[f"mean_{k}"]) for k in
           ("loss_G", "gan_A2B", "gan_B2A", "cyc_A", "cyc_B", "idt_A", "idt_B", "loss_D_A", "loss_D_B")}
    for k in ("grad_norm_G", "grad_norm_D"):
        out[f"{k}_mean"], out[f"{k}_p99"], out[f"{k}_max"] = (float(s[k].mean()), float(np.percentile(s[k], 99)),
                                                             float(s[k].max()))
    out["nonfinite_steps"] = int(summary["nonfinite_steps"])
    for k in ("params_total", "params_generators", "params_discriminators", "training_time_sec",
              "train_images_per_sec", "train_pairs_per_sec", "total_steps", "epochs"):
        out[k] = summary[k]
    out["training_time_min"] = summary["training_time_sec"] / 60
    out.update({k: v for k, v in summary.items() if k.startswith("peak_")})
    out["hardware"] = summary["hardware"].get("gpu_name") or summary["hardware"].get("cpu_model")
    out["gpu_at_start"] = summary.get("gpu_at_start", "")
    return out


@torch.no_grad()
def cycle_and_lpips(G, G_back, paths: list[str], size: int, device, lpips_fn, bs: int = 16) -> dict:
    tf = eval_transform(size)
    cyc, lp_tr, lp_rec, xs, ys, rs = [], [], [], [], [], []
    for i in range(0, len(paths), bs):
        x = torch.stack([load_image(p, tf) for p in paths[i:i + bs]]).to(device)
        y = G(x)
        r = G_back(y)
        cyc.append(((r - x).abs().mean((1, 2, 3)) / 2).cpu().numpy())   # [0,1] pixel units
        lp_tr.append(lpips_fn(x, y))
        lp_rec.append(lpips_fn(x, r))
        if i == 0:
            xs, ys, rs = x[:8].cpu(), y[:8].cpu(), r[:8].cpu()
    return {"cycle_l1": np.concatenate(cyc), "lpips_translation": np.concatenate(lp_tr),
            "lpips_reconstruction": np.concatenate(lp_rec), "examples": (xs, ys, rs)}


def evaluate(cfg: dict, run_id: str, promote: bool | None = None) -> dict:
    dirs = run_dirs(run_id)
    log = get_logger(f"eval.{run_id}", dirs["raw_logs"] / "eval.log")
    device = get_device(cfg.get("device", "auto"))
    e, d = cfg["eval"], cfg["data"]
    n_eval, pretrained = e.get("n_eval", 300), e.get("pretrained", True)
    set_seed(cfg["seed"])
    pred_root = repo_path(cfg["export"]["out_dir"])
    info = read_json(pred_root / "predictions_info.json")
    if info["run_id"] != run_id:
        raise SystemExit(f"{rel(pred_root)} holds predictions of {info['run_id']}, not {run_id}; run translate first")
    out = dirs["outputs"]
    log.info(f"EVAL run {run_id} n_eval={n_eval} pretrained_metric_nets={pretrained} device={device}")

    incep = InceptionFeatures(device, pretrained=pretrained)
    lpips_fn = LPIPSMetric(device, pretrained=pretrained)
    ck = run_dirs(run_id)["checkpoints"]
    gens = {"G_AB": load_generator(ck / "G_AB.pt", device), "G_BA": load_generator(ck / "G_BA.pt", device)}

    metrics, per_image, examples = {}, {}, {}
    feats_cache = {}
    for name, (src_key, pred, tgt_key, g, g_back) in DIRECTIONS.items():
        real_paths = list_images(repo_path(d[tgt_key]))[:n_eval]
        gen_paths = list_images(pred_root / pred)[:n_eval]
        src_paths = list_images(repo_path(d[src_key]))[:len(gen_paths)]
        for key, paths in (("real_" + tgt_key, real_paths), ("src_" + src_key, src_paths)):
            if key not in feats_cache:
                feats_cache[key] = incep.from_paths(paths)
        real, src, gen = feats_cache["real_" + tgt_key], feats_cache["src_" + src_key], incep.from_paths(gen_paths)
        fid, mifid = instructor_fid_mifid(real, gen)
        kid_m, kid_s = kid(real, gen, e.get("kid_subsets", 100), e.get("kid_subset_size", 100), seed=cfg["seed"])
        prdc = precision_recall_density_coverage(real, gen)
        cl = cycle_and_lpips(gens[g], gens[g_back], src_paths, d["crop_size"], device, lpips_fn)
        content_cos = paired_cosine_similarity(src, gen)
        g_n = gen / np.linalg.norm(gen, axis=1, keepdims=True)
        s_n = src / np.linalg.norm(src, axis=1, keepdims=True)
        nn_dist = torch.cdist(torch.from_numpy(gen).double(), torch.from_numpy(real).double()).min(1).values.numpy()
        metrics[name] = {
            "n_real": len(real), "n_generated": len(gen), "fid": fid, "mifid_script": mifid, "kid": kid_m,
            "kid_std": kid_s, **prdc, "cycle_l1": float(cl["cycle_l1"].mean()),
            "lpips_input_vs_translation": float(cl["lpips_translation"].mean()),
            "lpips_input_vs_reconstruction": float(cl["lpips_reconstruction"].mean()),
            "content_cosine_similarity": content_cos}
        per_image[name] = {"paths": src_paths, "gen_paths": gen_paths, "cycle_l1": cl["cycle_l1"],
                           "lpips": cl["lpips_translation"], "content_cos": (g_n * s_n).sum(1), "nn_dist": nn_dist}
        examples[name] = cl["examples"]
        log.info(f"{name}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                         for k, v in metrics[name].items()))

    fid_avg = (metrics["A2B (monet->photo)"]["fid"] + metrics["B2A (photo->monet)"]["fid"]) / 2
    mifid_avg = (metrics["A2B (monet->photo)"]["mifid_script"] + metrics["B2A (photo->monet)"]["mifid_script"]) / 2
    sub_rows = [{"ID": 1, "FID": fid_avg, "MiFID": mifid_avg}]
    smoke = cfg.get("smoke", False)
    sub_path = (out if smoke else MEMBER_DIR) / "submission.csv"
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "FID", "MiFID"])
        w.writeheader()
        w.writerows(sub_rows)
    log.info(f"submission: FID={fid_avg:.4f} MiFID={mifid_avg:.4f} -> {rel(sub_path)}")

    summary = read_json(out / "train_summary.json")
    tstats = training_stats(dirs["raw_logs"], summary)
    plot_training(dirs["raw_logs"], out)
    for name, (xs, ys, rs) in examples.items():
        tag = "B2A" if name.startswith("B2A") else "A2B"
        save_grid([xs, ys, rs], out / f"final_samples_{tag}.png")

    audit_path = pred_root / "human_audit" / "human_audit_results.json"
    audit = read_json(audit_path) if audit_path.exists() else None
    kag_path = MEMBER_DIR / "kaggle_leaderboard.json"
    kaggle = read_json(kag_path) if kag_path.exists() else {}

    # ---- failure candidates (worst cycle error, least content kept, least realistic) -------------------
    fc_lines = [f"# Visual failure candidates - run `{run_id}`", "",
                "Automatically ranked per direction; each grid row = input | translation | reconstruction.", ""]
    for name, pi in per_image.items():
        tag = "B2A" if name.startswith("B2A") else "A2B"
        g, g_back = DIRECTIONS[name][3], DIRECTIONS[name][4]
        for crit, order, label in (("cycle_l1", -1, "highest cycle-reconstruction error"),
                                   ("content_cos", 1, "lowest content cosine (content lost)"),
                                   ("nn_dist", -1, "farthest from any real target image (least realistic)")):
            idx = np.argsort(order * pi[crit])[:4]
            tf = eval_transform(d["crop_size"])
            x = torch.stack([load_image(pi["paths"][i], tf) for i in idx])
            with torch.no_grad():
                y = gens[g](x.to(device)).cpu()
                r = gens[g_back](y.to(device)).cpu()
            fname = f"failure_{tag}_{crit}.png"
            save_grid([x, y, r], out / fname)
            fc_lines.append(f"## {name} - {label}\n\n![{fname}]({fname})\n")
            for i in idx:
                fc_lines.append(f"- `{Path(pi['paths'][i]).name}` cycle L1 {pi['cycle_l1'][i]:.4f}, "
                                f"LPIPS {pi['lpips'][i]:.3f}, content cos {pi['content_cos'][i]:.3f}, "
                                f"NN distance {pi['nn_dist'][i]:.2f}")
            fc_lines.append("")
    (out / "failure_candidates.md").write_text("\n".join(fc_lines), encoding="utf-8")

    # ---- report ----------------------------------------------------------------------------------
    notes = {"fid": "Inception pool3, instructor protocol (lower is better)",
             "mifid_script": "instructor script's 'MiFID' = mean paired cosine distance",
             "kid": "unbiased MMD^2, cubic kernel (lower is better)", "kid_std": "std over KID subsets",
             "precision": "k=3 (realism)", "recall": "k=3 (diversity)", "density": "k=5", "coverage": "k=5",
             "cycle_l1": "mean |x - G_back(G(x))|, pixels in [0,1]",
             "lpips_input_vs_translation": "AlexNet LPIPS; higher = stronger change",
             "lpips_input_vs_reconstruction": "AlexNet LPIPS; lower = better cycle consistency",
             "content_cosine_similarity": "Inception features input vs translation; higher = content kept"}
    rows = [["Kaggle submission (avg of both directions)", "FID", fid_avg, "submission.csv"],
            ["Kaggle submission (avg of both directions)", "MiFID", mifid_avg, "submission.csv"]]
    for name, m in metrics.items():
        rows += [[name, k, v, notes.get(k, "")] for k, v in m.items()]
    rows += [["training", k, v, ""] for k, v in tstats.items()]
    if audit:
        rows.append(["human audit", "n_samples", audit["n_samples_scored_by_both"], "2 raters"])
        rows.append(["human audit", "overall_mean_score", audit["overall_mean_score"], "1-5"])
        for c, v in audit["criteria"].items():
            rows += [["human audit", f"{c}.{k}", val, ""] for k, val in v.items()]
    else:
        rows.append(["human audit", "status", "pending", "run src/human_audit.py make, rate, then score"])
    for k in ("team_name", "public_score", "private_score", "public_rank", "private_rank", "submitted_utc"):
        rows.append(["kaggle leaderboard", k, kaggle.get(k), "from kaggle_leaderboard.json (filled in by hand)"])
    rows.append(["run", "run_id", run_id, "evidence: reproducibility/raw_logs/... and outputs/<run_id>/"])
    for target in [out / "full_metrics_report.csv"] + ([] if smoke else [MEMBER_DIR / "full_metrics_report.csv",
                                                                        MEMBER_DIR / "metrics_report.csv"]):
        with open(target, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["section", "metric", "value", "notes"])
            w.writerows(rows)
    write_json(out / "eval_metrics.json", {"submission": sub_rows[0], "directions": metrics, "training": tstats,
                                          "human_audit": audit, "kaggle": kaggle})

    promote = (not smoke) if promote is None else promote
    if promote:
        _update_results(metrics, tstats, fid_avg, mifid_avg, audit, kaggle, run_id)
        fa = MEMBER_DIR / "failure_analysis.md"
        if not fa.exists() or AUTO_MARKER in fa.read_text(encoding="utf-8"):
            fa.write_text(f"{AUTO_MARKER}: generated by evaluate_local.py from run {run_id}. Look at the grids, describe "
                          "each failure in your own words, then delete this line. -->\n"
                          + "\n".join(fc_lines).replace("](failure_", f"](outputs/{run_id}/failure_"), encoding="utf-8")
    update_manifest(run_id, evaluation={"metrics": rel(out / "eval_metrics.json"), "submission": rel(sub_path),
                                        "full_metrics_report": rel(out / "full_metrics_report.csv"),
                                        "plots": [rel(p) for p in sorted(out.glob("*.png"))]})
    log.info(f"EVAL DONE -> {rel(out)}")
    return {"submission": sub_rows[0], "directions": metrics, "training": tstats}


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v))


def _update_results(metrics, tstats, fid_avg, mifid_avg, audit, kaggle, run_id) -> None:
    res = MEMBER_DIR / "results.md"
    if not res.exists():
        return
    names = list(metrics)
    keys = list(metrics[names[0]])
    lines = [f"_Auto-generated by `evaluate_local.py` from run `{run_id}` (source `outputs/{run_id}/full_metrics_report.csv`)._",
             "", f"**Kaggle submission (instructor protocol, average of both directions): FID = {fid_avg:.3f}, "
                 f"MiFID = {mifid_avg:.4f}**", "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {k} | " + " | ".join(_fmt(metrics[n][k]) for n in names) + " |" for k in keys]
    lines += ["", "| Training statistic | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in tstats.items() if k != "gpu_at_start"]
    lines += ["", f"GPU state at start of training: `{tstats.get('gpu_at_start', '')}`", ""]
    if audit:
        lines += ["| Human audit (2 raters, 30 samples) | mean | photo->Monet | Monet->photo | Cohen's kappa (quadratic) | exact agreement | within +-1 |",
                  "|---|---|---|---|---|---|---|"]
        for c, v in audit["criteria"].items():
            lines.append(f"| {c} | {v['mean_both']:.2f} | {v['mean_photo->monet']:.2f} | {v['mean_monet->photo']:.2f} | "
                         f"{v['cohen_kappa_quadratic']:.3f} | {v['agreement_exact']:.0%} | {v['agreement_within_1']:.0%} |")
    else:
        lines.append("_Human audit: pending (see `outputs/human_audit/INSTRUCTIONS.md`)._")
    lines += ["", "| Kaggle leaderboard | value |", "|---|---|"] + \
             [f"| {k} | {kaggle.get(k) if kaggle.get(k) is not None else 'to fill in kaggle_leaderboard.json'} |"
              for k in ("team_name", "public_score", "private_score", "public_rank", "private_rank")]
    text = res.read_text(encoding="utf-8")
    text = re.sub(r"(<!-- METRICS:START -->\n).*?(\n<!-- METRICS:END -->)", lambda m: m.group(1) + "\n".join(lines) + m.group(2),
                  text, flags=re.S)
    text = re.sub(r"outputs/(?:RUN_ID|[^/()\s]+)/(?=[a-z_A-Z0-9]+\.png)", f"outputs/{run_id}/", text)
    res.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="task3_gan/shibin_thomas/configs/cyclegan_v1.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--run-id", default="latest")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    evaluate(cfg, latest_run_id(cfg["run_name"]) if a.run_id == "latest" else a.run_id)


if __name__ == "__main__":
    main()
