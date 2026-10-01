"""Shared helpers: repo-relative paths, config loading, logging, seeding, manifests."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parent
MEMBER_DIR = SRC_DIR.parent                      # task1_llm/shibin_thomas
TASK_DIR = MEMBER_DIR.parent                     # task1_llm
REPO_ROOT = TASK_DIR.parent
MEMBER = MEMBER_DIR.name

if str(TASK_DIR) not in sys.path:                # makes `shared_eval` importable
    sys.path.insert(0, str(TASK_DIR))


# --------------------------------------------------------------------------- paths
def repo_path(p: str | Path) -> Path:
    """Resolve a path from the config. Relative paths are relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def rel(p: str | Path) -> str:
    """Path relative to the repo root, for logs (never log personal absolute paths)."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def run_dirs(run_id: str) -> dict[str, Path]:
    d = {
        "checkpoints": MEMBER_DIR / "checkpoints" / run_id,
        "outputs": MEMBER_DIR / "outputs" / run_id,
        "raw_logs": REPO_ROOT / "reproducibility" / "raw_logs" / TASK_DIR.name / MEMBER / run_id,
        "manifests": REPO_ROOT / "reproducibility" / "manifests" / TASK_DIR.name / MEMBER,
    }
    for v in d.values():
        v.mkdir(parents=True, exist_ok=True)
    return d


def latest_run_id(prefix: str | None = None) -> str:
    runs = sorted(
        (p for p in (MEMBER_DIR / "checkpoints").iterdir() if p.is_dir() and (p / "last_full.pt").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if prefix:
        runs = [p for p in runs if p.name.startswith(prefix)]
    if not runs:
        raise FileNotFoundError("no run with a last_full.pt checkpoint found")
    return runs[-1].name


# --------------------------------------------------------------------------- config
def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config; `overrides` are 'a.b=value' strings (value parsed as YAML)."""
    with open(repo_path(path)) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(val))
    cfg["_config_path"] = rel(repo_path(path))
    cfg["_overrides"] = list(overrides or [])
    return cfg


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- runtime
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(pref: str, device: torch.device) -> str:
    if pref != "auto":
        return pref
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def autocast_ctx(device: torch.device, precision: str):
    if precision == "fp32" or device.type not in ("cuda", "cpu"):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_mb(device: torch.device) -> dict[str, float]:
    try:
        import resource                                       # Linux / macOS
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out = {"peak_cpu_rss_mb": rss / (2**20 if sys.platform == "darwin" else 1024)}
    except ImportError:                                       # Windows
        import psutil
        out = {"peak_cpu_rss_mb": psutil.Process().memory_info().peak_wset / 2**20}
    if device.type == "cuda":
        out["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        out["peak_gpu_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    return out


def get_logger(name: str, log_file: Path) -> logging.Logger:
    """Logger writing to stdout and APPENDING to the raw log file (never truncates)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# --------------------------------------------------------------------------- manifests
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT, timeout=60).stdout.strip()
    except Exception:
        return ""


def git_info() -> dict[str, Any]:
    return {
        "commit": _cmd(["git", "rev-parse", "HEAD"]) or None,
        "branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or None,
        "dirty": bool(_cmd(["git", "status", "--porcelain"])),
    }


def hardware_info(device: torch.device) -> dict[str, Any]:
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), cpu)
    except OSError:
        pass
    info: dict[str, Any] = {
        "device": str(device),
        "cpu_model": cpu,
        "cpu_count": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(gpu_name=p.name, gpu_count=torch.cuda.device_count(),
                    gpu_total_memory_gb=round(p.total_memory / 2**30, 2),
                    cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version())
    return info


def environment_info() -> dict[str, Any]:
    freeze = _cmd([sys.executable, "-m", "pip", "freeze"]).splitlines()
    freeze = [l for l in freeze if "file://" not in l]            # drop local paths
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pip_freeze": freeze,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def update_manifest(run_id: str, **sections: Any) -> Path:
    path = run_dirs(run_id)["manifests"] / f"{run_id}.json"
    manifest = read_json(path) if path.exists() else {"run_id": run_id, "task": TASK_DIR.name, "member": MEMBER}
    for k, v in sections.items():
        manifest[k] = v
    manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(path, manifest)
    return path


def deep_copy_config(cfg: dict) -> dict:
    return copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
