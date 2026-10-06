#!/usr/bin/env python
"""
preprocess.py
-------------
Converts the Kaggle LiTS "archive" folder (lits_df / lits_train / lits_probe /
lits_test CSVs + dataset_6/ folder) into three compact files:

    <out_dir>/train.npz   (images uint8 [N,S,S], masks uint8 [N,S,S] in {0,1})
    <out_dir>/val.npz     (from lits_probe.csv)
    <out_dir>/test.npz    (from lits_test.csv)
    <out_dir>/meta.json

Why .npz instead of thousands of PNGs?  HPC parallel filesystems (Lustre/GPFS)
have file-count (inode) quotas and are slow with many tiny files. Three big
arrays load in seconds and the whole dataset sits in RAM during training.

The script is deliberately defensive about the dataset layout:
  * CSV columns for image / mask / slice index are auto-detected
    (override with --img_col / --mask_col / --slice_col).
  * Paths inside the CSV (often written for Kaggle, e.g. "../input/...") are
    re-resolved against --data_root by matching path suffixes.
  * Rows may point to 3-D NIfTI volumes OR single 2-D slices (.nii/.npy/.png).
  * CT intensity handling is auto-detected (raw HU -> liver window; already
    0-255 or 0-1 -> just rescaled). Override with --intensity.
  * If there are no usable CSVs, --no_csv falls back to discovering
    volume-N.nii / segmentation-N.nii pairs and splitting by patient.

Run `python inspect_dataset.py --data_root <archive>` first to see what it detects.
"""
import argparse
import functools
import json
import os
import random
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

SLICE_COL_NAMES = {"slice", "slice_idx", "slice_index", "slice_id", "slice_no",
                   "sliceno", "slice_num", "slice_number", "z"}


# --------------------------------------------------------------------------- #
# Array loading helpers
# --------------------------------------------------------------------------- #
def load_array(path):
    pl = path.lower()
    if pl.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        arr = np.asarray(nib.load(path).get_fdata(dtype=np.float32))
    elif pl.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.asarray(Image.open(path)).astype(np.float32)
        if arr.ndim == 3:            # RGB(A) png/jpg -> take first channel
            arr = arr[..., 0]
    return np.squeeze(arr)


@functools.lru_cache(maxsize=2)
def load_array_cached(path):
    """Small per-process cache: many CSV rows can point at the same volume."""
    return load_array(path)


def slices_last(a):
    """For 3-D arrays, move the slice axis (the odd one out) to the end."""
    if a.ndim != 3:
        return a
    s = a.shape
    if s[0] == s[1] and s[2] != s[0]:
        ax = 2
    elif s[0] == s[2] and s[1] != s[0]:
        ax = 1
    elif s[1] == s[2] and s[0] != s[1]:
        ax = 0
    else:
        ax = 2
    return np.moveaxis(a, ax, -1)


def to_uint8(slice2d, mode, level, width):
    if mode == "window":
        lo, hi = level - width / 2.0, level + width / 2.0
        x = (np.clip(slice2d, lo, hi) - lo) / (hi - lo) * 255.0
    elif mode == "unit":
        x = np.clip(slice2d, 0.0, 1.0) * 255.0
    else:  # "byte"
        x = np.clip(slice2d, 0.0, 255.0)
    return x.astype(np.uint8)


def resize_img(a_u8, size):
    return np.asarray(Image.fromarray(a_u8).resize((size, size), Image.BILINEAR))


def resize_mask(m01, size):
    im = Image.fromarray((m01.astype(np.uint8)) * 255).resize((size, size), Image.BILINEAR)
    return (np.asarray(im) > 127).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Path resolution (CSV paths -> real paths under --data_root)
# --------------------------------------------------------------------------- #
class PathResolver:
    def __init__(self, root):
        self.root = root
        self._index = None  # lazily built: {n_parts: {tail: path or None(ambiguous)}}

    def _build_index(self):
        print("  [resolver] direct paths missed; indexing files under", self.root, flush=True)
        idx = {n: {} for n in (1, 2, 3, 4)}
        count = 0
        for dp, _, fns in os.walk(self.root):
            for fn in fns:
                full = os.path.join(dp, fn)
                parts = full.replace("\\", "/").split("/")
                for n in idx:
                    key = "/".join(parts[-n:])
                    if key in idx[n]:
                        idx[n][key] = None      # ambiguous
                    else:
                        idx[n][key] = full
                count += 1
        print(f"  [resolver] indexed {count:,} files", flush=True)
        self._index = idx

    def resolve(self, raw):
        raw = str(raw).strip().replace("\\", "/")
        if os.path.exists(raw):
            return raw
        # 1) strip everything up to and including a "dataset_6/" marker
        if "dataset_6/" in raw:
            rest = raw.split("dataset_6/", 1)[1]
            for cand in (os.path.join(self.root, rest),
                         os.path.join(self.root, "dataset_6", rest),
                         os.path.join(self.root, "dataset_6", "dataset_6", rest)):
                if os.path.exists(cand):
                    return cand
        # 2) longest unique path-suffix match
        if self._index is None:
            self._build_index()
        parts = raw.split("/")
        for n in (4, 3, 2, 1):
            if len(parts) < n:
                continue
            hit = self._index[n].get("/".join(parts[-n:]))
            if hit:
                return hit
        return None


# --------------------------------------------------------------------------- #
# CSV column detection
# --------------------------------------------------------------------------- #
def detect_columns(df, img_col=None, mask_col=None, slice_col=None):
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}

    if img_col is None:
        cand = [c for c in cols if not any(k in low[c] for k in ("mask", "seg", "label", "tumor"))
                and any(k in low[c] for k in ("file", "path", "image", "img", "ct", "volume"))]
        img_col = cand[0] if cand else None
    if mask_col is None:
        # prefer a *liver* mask column, else any mask/seg/label column that isn't tumor
        cand = [c for c in cols if "liver" in low[c] and any(k in low[c] for k in ("mask", "seg", "path", "label"))]
        if not cand:
            cand = [c for c in cols if "tumor" not in low[c] and "lesion" not in low[c]
                    and any(k in low[c] for k in ("mask", "seg", "label"))]
        mask_col = cand[0] if cand else None
    if slice_col is None:
        cand = [c for c in cols if low[c] in SLICE_COL_NAMES]
        slice_col = cand[0] if cand else None

    if img_col is None or mask_col is None:
        raise RuntimeError(
            f"Could not auto-detect image/mask columns. CSV columns are: {cols}\n"
            f"Re-run with --img_col <name> --mask_col <name>.")
    return img_col, mask_col, slice_col


def build_tasks(csv_path, resolver, args):
    df = pd.read_csv(csv_path)
    if args.limit_rows:
        df = df.iloc[: args.limit_rows]
    img_col, mask_col, slice_col = detect_columns(df, args.img_col, args.mask_col, args.slice_col)
    print(f"  {os.path.basename(csv_path)}: {len(df):,} rows | image col='{img_col}' "
          f"mask col='{mask_col}' slice col={slice_col!r}", flush=True)
    tasks, missing = [], []
    for _, r in df.iterrows():
        ip, mp = resolver.resolve(r[img_col]), resolver.resolve(r[mask_col])
        if ip is None or mp is None:
            missing.append((r[img_col], r[mask_col]))
            continue
        sl = int(r[slice_col]) if slice_col is not None and pd.notna(r[slice_col]) else None
        tasks.append((ip, mp, sl))
    if missing:
        print(f"  WARNING: {len(missing):,} rows could not be resolved to files. First few:")
        for m in missing[:3]:
            print("    ", m)
    if not tasks:
        raise RuntimeError("No CSV rows resolved to real files. Check --data_root (it should be "
                           "the folder containing dataset_6/).")
    return tasks


def discover_pairs(root):
    """Fallback: find volume-N.nii / segmentation-N.nii pairs anywhere under root."""
    import glob
    vols = glob.glob(os.path.join(root, "**", "volume-*.nii*"), recursive=True)
    segs = glob.glob(os.path.join(root, "**", "segmentation-*.nii*"), recursive=True)
    idx = lambda p: int(re.search(r"-(\d+)\.nii", os.path.basename(p)).group(1))
    seg_by = {idx(s): s for s in segs}
    pairs = sorted((idx(v), v, seg_by[idx(v)]) for v in vols if idx(v) in seg_by)
    return pairs


# --------------------------------------------------------------------------- #
# Per-row worker
# --------------------------------------------------------------------------- #
def process_task(task, cfg):
    ip, mp, sl = task
    try:
        img = slices_last(load_array_cached(ip))
        msk = slices_last(load_array_cached(mp))
    except Exception as e:  # corrupt/unreadable file
        return ("err", f"{ip}: {e}")
    if img.shape != msk.shape:
        return ("err", f"shape mismatch {img.shape} vs {msk.shape} for {ip}")

    if img.ndim == 2:
        idxs, get = [0], (lambda a, i: a)
    else:
        idxs = [sl] if (sl is not None and 0 <= sl < img.shape[2]) else range(img.shape[2])
        get = lambda a, i: a[:, :, i]

    out_i, out_m = [], []
    for i in idxs:
        m = get(msk, i)
        m = (m == 1) if cfg["liver_only"] else (m > 0)
        if m.sum() < cfg["min_liver_pixels"]:
            continue
        im8 = to_uint8(get(img, i), cfg["intensity"], cfg["level"], cfg["width"])
        out_i.append(resize_img(im8, cfg["size"]))
        out_m.append(resize_mask(m, cfg["size"]))
    return ("ok", out_i, out_m)


def run_tasks(tasks, cfg, workers, name):
    t0 = time.time()
    imgs, masks, errs = [], [], []
    fn = functools.partial(process_task, cfg=cfg)
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap(fn, tasks, chunksize=32), 1):
            if res[0] == "err":
                errs.append(res[1])
            else:
                imgs.extend(res[1])
                masks.extend(res[2])
            if k % 5000 == 0 or k == len(tasks):
                print(f"  [{name}] {k:,}/{len(tasks):,} rows | {len(imgs):,} slices kept | "
                      f"{time.time()-t0:.0f}s", flush=True)
    if errs:
        print(f"  WARNING [{name}]: {len(errs)} unreadable rows (first: {errs[0]})")
    if not imgs:
        raise RuntimeError(f"No slices kept for split '{name}'.")
    return np.stack(imgs).astype(np.uint8), np.stack(masks).astype(np.uint8)


def detect_intensity(tasks, n=20):
    """Look at a sample of files once so every split uses the same rule."""
    step = max(1, len(tasks) // n)
    mins, maxs = [], []
    for ip, _, _ in tasks[::step][:n]:
        a = load_array(ip)
        mins.append(float(a.min()))
        maxs.append(float(a.max()))
    gmin, gmax = min(mins), max(maxs)
    if gmin >= -1e-4 and gmax <= 1.0001:
        mode = "unit"
    elif gmin >= 0 and gmax <= 255:
        mode = "byte"
    else:
        mode = "window"
    print(f"  intensity sample: min={gmin:.1f} max={gmax:.1f} -> mode '{mode}'", flush=True)
    if mode == "window" and gmin >= 0:
        print("  NOTE: values are all >= 0 but exceed 255 (shifted HU?). The default liver window "
              "(level 40/width 400) may be wrong -- check with inspect_dataset.py and use "
              "--hu_window if needed.")
    return mode


def save_split(out_dir, name, imgs, masks):
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, images=imgs, masks=masks)
    print(f"  saved {path}  images {imgs.shape}  liver fraction "
          f"{masks.mean()*100:.1f}%  ({os.path.getsize(path)/1e6:.0f} MB)", flush=True)
    return int(imgs.shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="the 'archive' folder (contains the CSVs and dataset_6/)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--csv_dir", default=None, help="folder with the CSVs (default: data_root)")
    ap.add_argument("--train_csv", default="lits_train.csv")
    ap.add_argument("--val_csv", default="lits_probe.csv")
    ap.add_argument("--test_csv", default="lits_test.csv")
    ap.add_argument("--img_col", default=None)
    ap.add_argument("--mask_col", default=None)
    ap.add_argument("--slice_col", default=None)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--intensity", choices=["auto", "window", "unit", "byte"], default="auto")
    ap.add_argument("--hu_window", type=float, nargs=2, default=[40, 400], metavar=("LEVEL", "WIDTH"))
    ap.add_argument("--min_liver_pixels", type=int, default=50)
    ap.add_argument("--liver_only", action="store_true", help="label==1 only (exclude tumor) instead of label>0")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    ap.add_argument("--limit_rows", type=int, default=0, help="debug: only use the first N rows per CSV")
    ap.add_argument("--no_csv", action="store_true", help="ignore CSVs; discover volume-N/segmentation-N pairs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_dir = args.csv_dir or args.data_root
    resolver = PathResolver(args.data_root)
    print(f"data_root={args.data_root}\nworkers={args.workers}", flush=True)

    split_tasks = {}
    if not args.no_csv and os.path.exists(os.path.join(csv_dir, args.train_csv)):
        print("Reading CSVs ...")
        split_tasks["train"] = build_tasks(os.path.join(csv_dir, args.train_csv), resolver, args)
        vp = os.path.join(csv_dir, args.val_csv)
        if os.path.exists(vp):
            split_tasks["val"] = build_tasks(vp, resolver, args)
        else:
            n_val = max(1, int(0.1 * len(split_tasks["train"])))
            print(f"  no {args.val_csv}: using the LAST {n_val} train rows as validation (unshuffled, "
                  f"so neighbouring slices/patients stay together)")
            split_tasks["val"] = split_tasks["train"][-n_val:]
            split_tasks["train"] = split_tasks["train"][:-n_val]
        tp = os.path.join(csv_dir, args.test_csv)
        if os.path.exists(tp):
            split_tasks["test"] = build_tasks(tp, resolver, args)
        else:
            raise RuntimeError(f"{tp} not found")
    else:
        print("Discovering volume/segmentation pairs (patient-level 80/10/10 split) ...")
        pairs = discover_pairs(args.data_root)
        if not pairs:
            raise RuntimeError("No lits CSVs and no volume-N.nii / segmentation-N.nii pairs found.")
        random.Random(args.seed).shuffle(pairs)
        n = len(pairs)
        n_te, n_va = max(1, n // 10), max(1, n // 10)
        sel = {"test": pairs[:n_te], "val": pairs[n_te:n_te + n_va], "train": pairs[n_te + n_va:]}
        split_tasks = {k: [(v, s, None) for _, v, s in lst] for k, lst in sel.items()}

    mode = args.intensity
    if mode == "auto":
        mode = detect_intensity(split_tasks["train"])
    cfg = dict(size=args.size, intensity=mode, level=args.hu_window[0], width=args.hu_window[1],
               min_liver_pixels=args.min_liver_pixels, liver_only=args.liver_only)

    counts = {}
    for name in ("train", "val", "test"):
        print(f"Processing {name} ...", flush=True)
        imgs, masks = run_tasks(split_tasks[name], cfg, args.workers, name)
        counts[name] = save_split(args.out_dir, name, imgs, masks)

    meta = dict(cfg=cfg, counts=counts, source=args.data_root, args=vars(args))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done.", counts)


if __name__ == "__main__":
    main()
