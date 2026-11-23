"""Torch side of the NR-IQA data pipeline: transforms, Dataset, DataLoaders.

Design choices (all deliberate for IQA, unlike the detection pipeline):
  * No resize to 256 by default -- resizing changes blur/noise/compression, i.e. the very
    thing being scored. Train on random native-resolution crops instead.
  * Test time: a deterministic grid of crops per image, predictions averaged.
  * No blur/JPEG augmentation -- it would change the label.
"""
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .records import ScoreNorm, get_splits, load_records

ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}
STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}


def stats_for(arch: str):
    key = "imagenet" if arch.lower().startswith("imagenet") else "clip"
    return MEAN[key], STD[key]


def grid_offsets(w: int, h: int, crop: int, grid: int):
    """Top-left (x, y) of a grid x grid set of evenly spaced crops. grid=1 -> center crop."""
    if grid <= 1:
        return [((w - crop) // 2, (h - crop) // 2)]
    xs = np.linspace(0, w - crop, grid).round().astype(int)
    ys = np.linspace(0, h - crop, grid).round().astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]


def _resize_short(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    if min(w, h) == short:
        return img
    scale = short / min(w, h)
    return img.resize((max(short, round(w * scale)), max(short, round(h * scale))), Image.BICUBIC)


class PrepareImage:
    """Optional short-side resize (0 = keep native), and upscale if smaller than the crop."""

    def __init__(self, crop: int, resize_short: int = 0):
        self.crop, self.resize_short = crop, resize_short

    def __call__(self, img):
        if self.resize_short:
            img = _resize_short(img, self.resize_short)
        if min(img.size) < self.crop:
            img = _resize_short(img, self.crop)
        return img


class GridCrops:
    """PIL image -> tensor [grid*grid, 3, crop, crop]."""

    def __init__(self, crop, grid, mean, std, resize_short=0):
        self.prep, self.crop, self.grid = PrepareImage(crop, resize_short), crop, grid
        self.mean, self.std = mean, std

    def __call__(self, img):
        img = self.prep(img)
        w, h = img.size
        crops = [TF.normalize(TF.to_tensor(TF.crop(img, y, x, self.crop, self.crop)), self.mean, self.std)
                 for x, y in grid_offsets(w, h, self.crop, self.grid)]
        return torch.stack(crops)


def build_train_transform(crop, mean, std, resize_short=0, flip=True):
    ops = [PrepareImage(crop, resize_short), T.RandomCrop(crop)]
    if flip:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class IQADataset(Dataset):
    """Returns (image_or_crops, score). Scores pass through `norm` if given (train only)."""

    def __init__(self, records, transform, norm: ScoreNorm = None):
        self.records, self.transform, self.norm = records, transform, norm

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r.path).convert("RGB")
        y = r.score if self.norm is None else self.norm(r.score)
        return self.transform(img), torch.tensor(y, dtype=torch.float32)


def build_loaders(args):
    """Returns dict(train=DataLoader, val={name: DataLoader}, test={name: DataLoader}, norms={name: dict}).

    train_sets: trained on (train split), validated on (val split), tested on (test split).
    eval_sets : extra datasets for testing. If also in train_sets -> its test split, else the whole
                dataset (cross-dataset evaluation).
    """
    mean, std = stats_for(args.arch)
    train_tf = build_train_transform(args.crop, mean, std, args.resize_short)
    eval_tf = GridCrops(args.crop, args.test_grid, mean, std, args.resize_short)
    ratios = tuple(args.split_ratios)
    split_csv = dict(kv.split('=', 1) for kv in args.split_csv)  # {'KonIQ-10k': '/path/to.csv'}

    train_parts, val_sets, test_sets, norms = [], {}, {}, {}
    for name in args.train_sets:
        recs = load_records(name, args.dataset_root, args.download)
        splits = get_splits(name, [r.key for r in recs], args.split_dir, args.split_seed, ratios,
                             split_csv=split_csv.get(name))
        by_key = {r.key: r for r in recs}
        pick = lambda part: [by_key[k] for k in splits[part]]
        norm = ScoreNorm.fit([r.score for r in pick("train")])
        norms[name] = norm.to_dict()
        train_parts.append(IQADataset(pick("train"), train_tf, norm))
        val_sets[name] = IQADataset(pick("val"), eval_tf)
        test_sets[name] = IQADataset(pick("test"), eval_tf)
        print(f"[data] {name}: train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}, "
              f"score range in train = [{norm.lo:g}, {norm.hi:g}]")

    for name in args.eval_sets:
        if name in test_sets:
            continue
        recs = load_records(name, args.dataset_root, args.download)
        test_sets[name] = IQADataset(recs, eval_tf)
        print(f"[data] {name}: {len(recs)} images (cross-dataset, whole set)")

    def loader(ds, train):
        bs = args.batch_size if train else args.eval_batch_size
        return DataLoader(ds, batch_size=bs, shuffle=train, drop_last=train,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0)

    return {
        "train": loader(ConcatDataset(train_parts), True),
        "val": {k: loader(v, False) for k, v in val_sets.items()},
        "test": {k: loader(v, False) for k, v in test_sets.items()},
        "norms": norms,
    }
