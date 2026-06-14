"""
Step (c): fine-tune an image classifier to flag scanned book pages that have a
foreign object on them (the defect = object occluding text). Objective: MINIMIZE
FALSE NEGATIVES — never miss a covered page — so we optimize recall on the `bad`
class and tune the decision threshold to favor catching defects.

- Backbone: EfficientNetV2-S (ImageNet pretrained), 384px input (small objects matter).
- Data: Data/_dataset/{train,val,test}/{good,bad}  (torchvision ImageFolder).
- Augmentation keeps the whole page in frame (Resize+affine, NO random crop) so a
  `bad` page never has its object cropped away (which would be a mislabeled sample).
- Threshold: pick on val for target recall on bad >= TARGET_RECALL, report precision.
- Eval: synthetic test split + the REAL phone-photo holdout (Data/_holdout_real/bad,
  all bad) -> the honest 'does it catch real objects' recall.
- Saves best model + chosen threshold to pipeline/fo_classifier_effv2s.pt.

Runs on the local RTX 5080 (CUDA).
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_V2_S_Weights

ROOT = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data")
DATA = ROOT / "_dataset"
HOLDOUT_BAD = ROOT / "_holdout_real" / "bad"
CKPT = Path(r"D:\E\Data Science Projects\ClaudeDoc\pipeline\fo_classifier_effv2s.pt")

IMG = 384
BATCH = 24
EPOCHS = 12
LR = 5e-4
TARGET_RECALL = 0.99          # on the bad class
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
device = "cuda" if torch.cuda.is_available() else "cpu"

# ImageFolder sorts classes alphabetically -> bad=0, good=1. Positive (defect) = bad.
BAD = 0

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08)),
    transforms.ColorJitter(0.15, 0.15, 0.10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def loader(split, tf, shuffle):
    ds = datasets.ImageFolder(DATA / split, transform=tf)
    return ds, DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=4,
                          pin_memory=True, persistent_workers=True)


@torch.no_grad()
def predict_probs(model, dl):
    model.eval()
    ps, ys = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        p_bad = torch.softmax(logits.float(), 1)[:, BAD].cpu().numpy()
        ps.append(p_bad)
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def recall_precision(p_bad, y_true, thr):
    pred_bad = p_bad >= thr
    is_bad = (y_true == BAD)
    tp = np.sum(pred_bad & is_bad)
    fn = np.sum(~pred_bad & is_bad)
    fp = np.sum(pred_bad & ~is_bad)
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    return recall, precision, tp, fn, fp


def pick_threshold(p_bad, y_true, target_recall):
    """Lowest-FP threshold that still achieves target recall on bad."""
    best_thr, best_prec = 0.5, -1
    for thr in np.linspace(0.01, 0.99, 99):
        r, prec, *_ = recall_precision(p_bad, y_true, thr)
        if r >= target_recall and prec > best_prec:
            best_prec, best_thr = prec, thr
    return best_thr


def main():
    print(f"device={device}")
    train_ds, train_dl = loader("train", train_tf, True)
    val_ds, val_dl = loader("val", eval_tf, False)
    test_ds, test_dl = loader("test", eval_tf, False)
    print("classes:", train_ds.classes, "(positive/defect = bad)")

    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 2)
    model = model.to(device)

    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.3, 1.0], device=device))  # nudge bad
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    best_val_acc, best_state = -1, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * x.size(0)
        sched.step()
        p, yv = predict_probs(model, val_dl)
        val_acc = np.mean((p >= 0.5) == (yv == BAD))
        r, prec, *_ = recall_precision(p, yv, 0.5)
        print(f"ep{ep:2d} loss={run/len(train_ds):.3f} val_acc={val_acc:.3f} "
              f"bad_recall@.5={r:.3f} bad_prec@.5={prec:.3f} ({time.time()-t0:.0f}s)")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # threshold tuned on val for target recall
    pv, yv = predict_probs(model, val_dl)
    thr = pick_threshold(pv, yv, TARGET_RECALL)
    vr, vp, *_ = recall_precision(pv, yv, thr)
    print(f"\nChosen threshold={thr:.2f} (val bad recall={vr:.3f}, precision={vp:.3f})")

    # synthetic test
    pt, yt = predict_probs(model, test_dl)
    tr, tp_, tp, fn, fp = recall_precision(pt, yt, thr)
    print(f"SYNTH TEST: bad recall={tr:.3f} precision={tp_:.3f}  "
          f"(missed bad pages FN={fn}, false alarms FP={fp})")

    # real phone-photo holdout (all bad) -> fraction caught at threshold
    if HOLDOUT_BAD.exists():
        hold_ds = datasets.ImageFolder(HOLDOUT_BAD.parent, transform=eval_tf)
        hold_dl = DataLoader(hold_ds, batch_size=BATCH, num_workers=4)
        ph, yh = predict_probs(model, hold_dl)
        bad_idx = hold_ds.classes.index("bad")
        mask = yh == bad_idx
        caught = np.mean(ph[mask] >= thr)
        print(f"REAL HOLDOUT: caught {caught*100:.1f}% of {mask.sum()} real "
              f"phone-photo bad pages (recall on real captures)")

    torch.save({"state_dict": model.state_dict(), "threshold": float(thr),
                "classes": train_ds.classes, "img": IMG,
                "arch": "efficientnet_v2_s"}, CKPT)
    print(f"\nsaved -> {CKPT}")


if __name__ == "__main__":
    main()
