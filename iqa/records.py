"""Torch-free dataset plumbing: read metadata from `iqadataset`, build/load splits.

Everything the rest of the code needs from `iqadataset` goes through
`load_records`, so if the package's return type is different from what
`_column` expects, this is the only place to touch.
"""
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Record:
    path: str     # path that can be opened from disk
    key: str      # stable id: path relative to dataset_root, '/' separated
    score: float  # raw subjective score (MOS) as stored by iqadataset


def _column(ds, name: str, idx: int, n_cols: int):
    """Pull one column out of whatever `iqadataset.load_dataset` returned."""
    if hasattr(ds, "columns") and name in getattr(ds, "columns"):  # pandas DataFrame
        return list(ds[name])
    if isinstance(ds, dict) and name in ds:                       # dict of lists
        return list(ds[name])
    if hasattr(ds, name):                                          # object with attributes
        return list(getattr(ds, name))
    if isinstance(ds, (list, tuple)) and ds:
        row = ds[0]
        if isinstance(row, dict) and name in row:                  # list of dict rows
            return [r[name] for r in ds]
        if isinstance(row, (list, tuple)) and len(row) == n_cols:  # list of tuples, in requested order
            return [r[idx] for r in ds]
    raise TypeError(
        f"Don't know how to read column '{name}' from a {type(ds)} returned by "
        f"iqadataset.load_dataset. Adapt `_column` in iqa/records.py. repr: {repr(ds)[:300]}"
    )


def load_records(name: str, dataset_root: str, download: bool = False) -> List[Record]:
    from iqadataset import load_dataset  # imported lazily so the rest is testable without it

    attrs = ["dis_img_path", "score"]
    ds = load_dataset(name, dataset_root=dataset_root, attributes=attrs, download=download)
    paths = _column(ds, "dis_img_path", 0, len(attrs))
    scores = _column(ds, "score", 1, len(attrs))
    if len(paths) != len(scores):
        raise ValueError(f"{name}: {len(paths)} paths vs {len(scores)} scores")

    records = []
    for p, s in zip(paths, scores):
        p = str(p)
        if os.path.isabs(p) or os.path.exists(p):
            full = p
        else:
            full = os.path.join(dataset_root, p)
        key = os.path.relpath(full, dataset_root)
        key = key.replace(os.sep, "/")
        s = float(s)
        if s != s:  # NaN
            raise ValueError(f"{name}: NaN score for {p}")
        records.append(Record(path=full, key=key, score=s))
    records.sort(key=lambda r: r.key)  # deterministic regardless of iqadataset's ordering
    if len({r.key for r in records}) != len(records):
        raise ValueError(f"{name}: duplicate image paths")
    return records


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def make_splits(keys: Sequence[str], ratios: Tuple[float, float, float], seed: int) -> Dict[str, List[str]]:
    keys = sorted(keys)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }


def get_splits(name: str, keys: Sequence[str], split_dir: str, seed: int,
               ratios: Tuple[float, float, float]) -> Dict[str, List[str]]:
    """Load the split file if it exists, otherwise create and save it.

    The file is plain JSON {"train": [key, ...], "val": [...], "test": [...]}.
    To use an official split, write a file with that name in the same format.
    """
    tag = "-".join(str(int(round(r * 100))) for r in ratios)
    path = os.path.join(split_dir, f"{name}_s{seed}_{tag}.json")
    if os.path.exists(path):
        with open(path) as f:
            splits = json.load(f)
        known = set(keys)
        for part in ("train", "val", "test"):
            missing = [k for k in splits.get(part, []) if k not in known]
            if missing:
                raise ValueError(f"{path}: {len(missing)} keys not in dataset, e.g. {missing[:3]}")
        return splits
    splits = make_splits(keys, ratios, seed)
    os.makedirs(split_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f)
    return splits


class ScoreNorm:
    """Affine map raw score -> [0, 1], fitted on the train split of one dataset."""

    def __init__(self, lo: float, hi: float):
        if hi <= lo:
            raise ValueError("degenerate score range")
        self.lo, self.hi = float(lo), float(hi)

    @classmethod
    def fit(cls, scores: Sequence[float]) -> "ScoreNorm":
        return cls(min(scores), max(scores))

    def __call__(self, s: float) -> float:
        return (float(s) - self.lo) / (self.hi - self.lo)

    def to_dict(self):
        return {"lo": self.lo, "hi": self.hi}
