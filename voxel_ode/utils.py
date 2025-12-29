import os
import re
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_run_index_from_name(name: str) -> int:
    """
    Extract the numeric run index from filenames like 'v2.4_0015.h5'.
    """
    base = os.path.basename(name)
    m = re.search(r"_(\d+)\.h5$", base)
    if not m:
        raise ValueError(f"Cannot parse run index from filename: {name}")
    return int(m.group(1))


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except Exception:
        return True


def ddp_print(*args, **kwargs):
    """Print only from main process with timestamp prefix."""
    if is_main_process():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # If first arg is a string, prepend timestamp
        if args and isinstance(args[0], str):
            new_args = (f"[{timestamp}] {args[0]}",) + args[1:]
            print(*new_args, **kwargs)
        else:
            print(f"[{timestamp}]", *args, **kwargs)


class Welford:
    """Streaming mean/variance computation using Welford's online algorithm."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None

    def update(self, x: np.ndarray):
        """Update statistics with new data."""
        x = x.astype(np.float64, copy=False).ravel()
        if self.mean is None:
            self.mean = np.zeros(1, dtype=np.float64)
            self.M2 = np.zeros(1, dtype=np.float64)
        for xi in x:
            self.n += 1
            delta = xi - self.mean[0]
            self.mean[0] += delta / self.n
            delta2 = xi - self.mean[0]
            self.M2[0] += delta * delta2

    def merge(self, other: "Welford"):
        """Merge statistics from another Welford instance."""
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean = other.mean.copy()
            self.M2 = other.M2.copy()
            return
        delta = other.mean[0] - self.mean[0]
        tot = self.n + other.n
        self.mean[0] = (self.n * self.mean[0] + other.mean[0] * other.n) / tot
        self.M2[0] += other.M2[0] + delta * delta * self.n * other.n / tot
        self.n = tot

    def finalize(self):
        """Return (mean, variance)."""
        if self.n < 2:
            return float(self.mean[0]) if self.mean is not None else 0.0, 0.0
        return float(self.mean[0]), float(self.M2[0] / (self.n - 1))
