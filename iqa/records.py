"""Torch-free dataset plumbing: read metadata from `iqadataset`, build/load splits.

Everything the rest of the code needs from `iqadataset` goes through
`load_records`; if the package's return type ever changes, `_rows` is the only place to touch.
"""
import csv
import json
import os
import posixpath
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Record:
    path: str     # path that can be opened from disk
    key: str      # stable id: path relative to dataset_root, '/' separated
    score: float  # raw subjective score (MOS) as stored by iqadataset


def _rows(ds, attrs: Sequence[str]) -> List[dict]:
    """Turn whatever `iqadataset.load_dataset` returned into a list of {attr: value} dicts.

    Confirmed shape (iqadataset 0.1.x): an `iqadataset.dataset.IQADataset` object where `ds[i]` is a
    dict such as {'dis_img_path': ..., 'score': ...} and no image is decoded. The other shapes are
    kept as cheap fallbacks in case the package changes.
    """
    if hasattr(ds, "to_dict") and hasattr(ds, "columns"):                      # pandas DataFrame
        return ds.to_dict("records")
    if isinstance(ds, dict) and all(a in ds for a in attrs):                   # dict of columns
        return [dict(zip(attrs, vals)) for vals in zip(*(ds[a] for a in attrs))]
    if isinstance(ds, (list, tuple)):                                          # list of dicts / tuples
        if ds and isinstance(ds[0], dict):
            return list(ds)
        return [dict(zip(attrs, row)) for row in ds]
    if hasattr(ds, "__getitem__"):                                             # IQADataset: ds[i] -> dict
        n = len(ds) if hasattr(ds, "__len__") else 10 ** 7
        rows = []
        try:
            for i in range(n):
                rows.append(ds[i])
        except IndexError:
            pass
        if rows and isinstance(rows[0], dict):
            return rows
    raise TypeError(f"Don't know how to read rows from a {type(ds)} returned by iqadataset.load_dataset. "
                    f"Adapt `_rows` in iqa/records.py. repr: {repr(ds)[:300]}")


def load_records(name: str, dataset_root: str, download: bool = False) -> List[Record]:
    from iqadataset import load_dataset  # imported lazily so the rest is testable without it

    attrs = ["dis_img_path", "score"]
    ds = load_dataset(name, dataset_root=dataset_root, attributes=attrs, download=download)
    rows = _rows(ds, attrs)
    if not rows:
        raise ValueError(f"{name}: iqadataset returned no rows (dataset_root={dataset_root!r}, download={download})")

    records = []
    for row in rows:
        p, s = str(row["dis_img_path"]), row["score"]
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


_SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "validation": "val",
                  "test": "test", "testing": "test"}


def splits_from_csv(path: str, keys: Sequence[str], name_col: str = "image_name",
                    split_col: str = "set") -> Dict[str, List[str]]:
    """Official split from a CSV with one row per image (e.g. KonIQ's koniq10k_distributions_sets.csv).

    Rows are matched to dataset keys by file name, so the CSV doesn't need the folder structure.
    """
    by_base: Dict[str, List[str]] = {}
    for k in keys:
        by_base.setdefault(posixpath.basename(k), []).append(k)
    clash = [b for b, v in by_base.items() if len(v) > 1]
    if clash:
        raise ValueError(f"file names are not unique in this dataset (e.g. {clash[:3]}); can't match by name")

    splits = {"train": [], "val": [], "test": []}
    unknown, unmatched = set(), []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for col in (name_col, split_col):
            if col not in (reader.fieldnames or []):
                raise ValueError(f"{path}: no column '{col}', found {reader.fieldnames}")
        for row in reader:
            part = _SPLIT_ALIASES.get(row[split_col].strip().lower())
            base = row[name_col].strip()
            if part is None:
                unknown.add(row[split_col])
            elif base not in by_base:
                unmatched.append(base)
            else:
                splits[part].append(by_base[base][0])
    if unknown:
        raise ValueError(f"{path}: unrecognised split labels {sorted(unknown)}")
    if unmatched:
        raise ValueError(f"{path}: {len(unmatched)} images not found in the dataset, e.g. {unmatched[:3]}")
    empty = [p for p, v in splits.items() if not v]
    if empty:
        raise ValueError(f"{path}: no images for split(s) {empty}")
    covered = sum(len(v) for v in splits.values())
    if covered != len(keys):
        print(f"[data] warning: split CSV covers {covered} of {len(keys)} images; the rest are unused")
    return {p: sorted(v) for p, v in splits.items()}


def get_splits(name: str, keys: Sequence[str], split_dir: str, seed: int,
               ratios: Tuple[float, float, float], split_csv: Optional[str] = None) -> Dict[str, List[str]]:
    """Official CSV split if `split_csv` is given; otherwise load the split file if it exists,
    or create and save a seeded random one.

    The saved file is plain JSON {"train": [key, ...], "val": [...], "test": [...]};
    to use any other split, write a file with that name in the same format.
    """
    if split_csv:
        return splits_from_csv(split_csv, keys)
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
