#!/usr/bin/env python
"""
train.py -- everything in one file: model + training + test evaluation.
  * resumes automatically from <out_dir>/last.pt
  * stops cleanly (exit code 10) before the walltime runs out, so run.sbatch can resubmit
  * when all epochs are done it evaluates on the test split and writes results/
"""
import argparse, csv, json, math, os, signal, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

# ------------------------------ model ------------------------------
class ResBlock(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.c1, s.c2, s.c3 = nn.Conv2d(i, o, 3, padding=1), nn.Conv2d(o, o, 3, padding=1), nn.Conv2d(o, o, 3, padding=1)
        s.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
    def forward(s, x):
        y = F.relu(s.c2(F.relu(s.c1(x))) + s.skip(x))
        return F.relu(s.c3(y))

class DecBlock(nn.Module):
    def __init__(s, i, skip, o):
        super().__init__()
        s.up = nn.ConvTranspose2d(i, o, 2, stride=2)
        s.c1, s.c2 = nn.Conv2d(o + skip, o, 3, padding=1), nn.Conv2d(o, o, 3, padding=1)
    def forward(s, x, skip):
        x = F.relu(s.up(x))
        x = torch.cat([skip, x], 1)
        return F.relu(s.c2(F.relu(s.c1(x))))

class ResUNet(nn.Module):
    """Modified ResUNet (3 residual encoder blocks, bridge, 3 decoder blocks). Returns LOGITS."""
    def __init__(s, base=32, drop=0.3):
        super().__init__()
        c1, c2, c3, cb = base, base * 2, base * 4, base * 8
        s.e1, s.e2, s.e3 = ResBlock(1, c1), ResBlock(c1, c2), ResBlock(c2, c3)
        s.drop = nn.Dropout2d(drop)
        s.b1, s.b2 = nn.Conv2d(c3, cb, 3, padding=1), nn.Conv2d(cb, cb, 3, padding=1)
        s.d3, s.d2, s.d1 = DecBlock(cb, c3, c3), DecBlock(c3, c2, c2), DecBlock(c2, c1, c1)
        s.h1, s.h2 = nn.Conv2d(c1, c1, 3, padding=1), nn.Conv2d(c1, 1, 1)
    def forward(s, x):
        e1 = s.e1(x)
        e2 = s.e2(F.max_pool2d(e1, 2))
        e3 = s.drop(s.e3(F.max_pool2d(e2, 2)))
        b = F.relu(s.b1(F.max_pool2d(e3, 2)))
        b = s.drop(F.relu(s.b2(b)))
        d = s.d1(s.d2(s.d3(b, e3), e2), e1)
        return s.h2(F.relu(s.h1(d)))

# ------------------------------ metrics / loss ------------------------------
def dice_loss(prob, y, eps=1e-7):
    prob, y = prob.flatten(1), y.flatten(1)
    return 1 - ((2 * (prob * y).sum(1) + eps) / (prob.sum(1) + y.sum(1) + eps)).mean()

@torch.no_grad()
def metrics(gt, pred, eps=1e-7):
    """Per-slice DSC, Accuracy, Jaccard, VOE, RVD in % (paper Eqs. 1-5)."""
    gt, pred = gt.flatten(1) > 0.5, pred.flatten(1) > 0.5
    tp, fp = (gt & pred).sum(1).float(), (~gt & pred).sum(1).float()
    fn, tn = (gt & ~pred).sum(1).float(), (~gt & ~pred).sum(1).float()
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return {"DSC": dice * 100, "Accuracy": (tp + tn) / (tp + tn + fp + fn + eps) * 100,
            "Jaccard": (tp + eps) / (tp + fp + fn + eps) * 100, "VOE": (1 - dice) * 100,
            "RVD": (pred.sum(1).float() - gt.sum(1).float()) / (gt.sum(1).float() + eps) * 100}

# ------------------------------ data ------------------------------
def load(data_dir, split, limit=0):
    d = np.load(os.path.join(data_dir, f"{split}.npz"))
    X, Y = d["images"], d["masks"]
    if limit: X, Y = X[:limit], Y[:limit]
    return torch.from_numpy(X), torch.from_numpy(Y)

def to_gpu(xb, yb, dev):
    return xb.to(dev).float().div_(255).unsqueeze(1), yb.to(dev).float().unsqueeze(1)

def batches(X, Y, bs, shuffle=False, drop_last=False, gen=None):
    n = X.size(0)
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    stop = (n // bs) * bs if drop_last else n
    for i in range(0, stop, bs):
        idx = order[i:i + bs]
        yield X[idx], Y[idx]

def random_rotate(x, y):
    """Paper augmentation: random rotation by 0,45,...,315 degrees, done on the GPU."""
    ang = torch.randint(8, (x.size(0),), device=x.device).float() * (math.pi / 4)
    c, s_, z = torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)
    theta = torch.stack([torch.stack([c, -s_, z], 1), torch.stack([s_, c, z], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    y = (F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False) > 0.5).float()
    return x, y

# ------------------------------ evaluation ------------------------------
def evaluate(model, data_dir, out_dir, dev):
    os.makedirs(out_dir, exist_ok=True)
    X, Y = load(data_dir, "test")
    model.eval()
    keys = ["DSC", "Accuracy", "Jaccard", "VOE", "RVD"]
    res = {k: [] for k in keys}
    pick = set(np.linspace(0, len(X) - 1, min(12, len(X))).astype(int).tolist())
    seen = 0
    with torch.no_grad():
        for xb, yb in batches(X, Y, 128):
            x, y = to_gpu(xb, yb, dev)
            pred = (torch.sigmoid(model(x).float()) > 0.5).float()
            m = metrics(y, pred)
            for k in keys: res[k].extend(m[k].cpu().tolist())
            for i in range(x.size(0)):
                if seen + i in pick:   # panel: input | truth | prediction
                    g = lambda a: np.stack([(a.cpu().numpy() * 255).astype(np.uint8)] * 3, -1)
                    Image.fromarray(np.concatenate([g(x[i, 0]), g(y[i, 0]), g(pred[i, 0])], 1)).save(
                        os.path.join(out_dir, f"sample_{seen + i:06d}.png"))
            seen += x.size(0)
    mean = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"\nTEST RESULTS on {len(X):,} slices (mean per slice, %):")
    for k in keys: print(f"  {k:9s} {mean[k]:7.2f}")
    json.dump({"n_slices": len(X), "mean": mean}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "metrics_per_slice.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for i in range(len(X)): w.writerow([f"{res[k][i]:.3f}" for k in keys])

# ------------------------------ training ------------------------------
STOP = False
def _sig(*_):
    global STOP; STOP = True

def save(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_decay", type=float, default=5e-3, help="x this every --drop_period epochs (paper: 5e-3; try 0.1)")
    ap.add_argument("--drop_period", type=int, default=50)
    ap.add_argument("--time_limit_min", type=float, default=0)
    ap.add_argument("--max_samples", type=int, default=0, help="debug: use first N slices only")
    args = ap.parse_args()

    t_start = time.time()
    signal.signal(signal.SIGTERM, _sig)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    print(f"device: {torch.cuda.get_device_name(0) if amp else 'CPU'} | torch {torch.__version__}", flush=True)

    Xtr, Ytr = load(args.data_dir, "train", args.max_samples)
    Xva, Yva = load(args.data_dir, "val", args.max_samples)
    print(f"train {len(Xtr):,} | val {len(Xva):,} slices", flush=True)

    model = ResUNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: args.lr_decay ** (e // args.drop_period))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    bce = nn.BCEWithLogitsLoss()

    last, best = os.path.join(args.out_dir, "last.pt"), os.path.join(args.out_dir, "best.pt")
    start, best_dice = 0, 0.0
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best_dice = ck["epoch"] + 1, ck["best"]
        print(f"RESUMED at epoch {start + 1}/{args.epochs} (best val Dice {best_dice*100:.2f}%)", flush=True)

    log = os.path.join(args.out_dir, "train_log.csv")
    if not os.path.exists(log):
        open(log, "w").write("epoch,train_loss,val_loss,val_dice,seconds\n")
    gen = torch.Generator().manual_seed(42)
    for _ in range(start): torch.randperm(len(Xtr), generator=gen)

    times = []
    for epoch in range(start, args.epochs):
        t0 = time.time(); model.train(); tot, n = 0.0, 0
        for xb, yb in batches(Xtr, Ytr, args.batch_size, True, True, gen):
            x, y = to_gpu(xb, yb, dev)
            x, y = random_rotate(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            logits = logits.float()
            loss = bce(logits, y) + dice_loss(torch.sigmoid(logits), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()

        model.eval(); vl, vd, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in batches(Xva, Yva, 128):
                x, y = to_gpu(xb, yb, dev)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    logits = model(x)
                logits = logits.float()
                vl += (bce(logits, y) + dice_loss(torch.sigmoid(logits), y)).item() * x.size(0)
                vd += metrics(y, (torch.sigmoid(logits) > 0.5).float())["DSC"].sum().item() / 100
                nv += x.size(0)
        vl, vd = vl / nv, vd / nv
        dt = time.time() - t0; times.append(dt)
        print(f"Epoch {epoch+1}/{args.epochs} | train {tot/n:.4f} | val {vl:.4f} | val_dice {vd*100:.2f}% | {dt:.0f}s", flush=True)
        open(log, "a").write(f"{epoch+1},{tot/n:.5f},{vl:.5f},{vd:.5f},{dt:.0f}\n")

        if vd > best_dice: best_dice, improved = vd, True
        else: improved = False
        ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  scaler=scaler.state_dict(), epoch=epoch, best=best_dice)
        save(ck, last)
        if improved: save(ck, best)

        if epoch + 1 < args.epochs:
            out_of_time = args.time_limit_min and \
                (time.time() - t_start) / 60 + 1.5 * np.mean(times[-5:]) / 60 > args.time_limit_min
            if STOP or out_of_time:
                print("Out of walltime -> checkpoint saved, exiting with code 10 (job will be resubmitted).", flush=True)
                return 10

    print(f"\nTraining complete. Best val Dice {best_dice*100:.2f}%", flush=True)
    model.load_state_dict(torch.load(best, map_location=dev, weights_only=False)["model"])
    evaluate(model, args.data_dir, os.path.join(args.out_dir, "results"), dev)
    open(os.path.join(args.out_dir, "DONE"), "w").write("done\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
