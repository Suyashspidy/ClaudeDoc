"""
Step 2 of the foreign-object-classifier pipeline: composite the extracted object
cutouts onto real scanned base pages to synthesize the BAD class in correct scanner
modality, and emit a balanced, leakage-safe train/val/test image dataset.

  good = clean scanned base page (no object)
  bad  = same modality base page + one real object composited over the text
         (random scale/rotation/position, soft drop-shadow, feathered edges)

Leakage control: base pages AND object cutouts are partitioned into disjoint
train/val/test pools (objects stratified by category). A page or object never
appears in two splits.

Layout (torchvision ImageFolder, ready for the ROCm ResNet/EfficientNet train):
  Data/_dataset/{train,val,test}/{good,bad}/*.jpg

Usage:
  python step2_composite.py --preview      # 24 sample composites -> preview sheet
  python step2_composite.py                # full generation
"""
import argparse
import hashlib
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data")
BASE_SRC = ROOT / "_base_src"
OBJ_DIR = ROOT / "_objects" / "cutouts"
PAGE_CACHE = ROOT / "_base_pages"
DATASET = ROOT / "_dataset"

TARGET_LONG = 1100           # longer side of cached base page (px)
INK_MIN, INK_MAX = 0.02, 0.35  # keep pages whose dark-pixel fraction is in this band
N_PER_CLASS = 4000           # good and bad each (full run)
MAX_KEPT_PER_BOOK = 150      # cap pages per book so no single book dominates
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 1234

# object placement / realism
SCALE_RANGE = (0.30, 0.80)   # object long side as fraction of page long side
ROT_RANGE = (-25, 25)
SHADOW_OFFSET = (6, 10)
SHADOW_BLUR = 9
SHADOW_OPACITY = (0.30, 0.55)


# ---------------------------------------------------------------- page cache
def render_page(page) -> Image.Image:
    zoom = TARGET_LONG / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def ink_fraction(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"))
    return float((g < 128).mean())


def build_page_cache() -> list[Path]:
    PAGE_CACHE.mkdir(parents=True, exist_ok=True)
    cached = sorted(PAGE_CACHE.glob("*.jpg"))
    if cached:
        print(f"page cache: {len(cached)} pages (reusing)")
        return cached
    kept, skipped = [], 0
    for pdf in sorted(BASE_SRC.glob("*.pdf")):
        doc = fitz.open(pdf)
        kept_this = 0
        for pno in range(len(doc)):
            if kept_this >= MAX_KEPT_PER_BOOK:
                break
            img = render_page(doc[pno])
            if not (INK_MIN < ink_fraction(img) < INK_MAX):
                skipped += 1
                continue
            out = PAGE_CACHE / f"{pdf.stem}__p{pno:04d}.jpg"
            img.save(out, quality=90)
            kept.append(out)
            kept_this += 1
        doc.close()
        print(f"  cached {pdf.stem}: {kept_this} kept (running total {len(kept)})")
    print(f"page cache built: {len(kept)} kept, {skipped} skipped (blank/plate)")
    return sorted(kept)


# ---------------------------------------------------------------- splitting
def assign_split(key: str) -> str:
    """Deterministic split from a stable hash of the item key."""
    h = int(hashlib.md5(f"{SEED}:{key}".encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < SPLITS["train"]:
        return "train"
    if h < SPLITS["train"] + SPLITS["val"]:
        return "val"
    return "test"


def book_of(page_path: Path) -> str:
    """Book stem from a cached page filename '<book>__p0000.jpg'."""
    return page_path.name.split("__p")[0]


def build_pools():
    pages = defaultdict(list)
    for p in build_page_cache():
        # Split by BOOK, not by page: every page of a book lands in the same split,
        # so val/test books are never seen in training. This is the honest eval that
        # the old page-level split lacked (which let the model memorise books).
        pages[assign_split(f"book:{book_of(p)}")].append(p)
    # objects stratified by category so every split has every category
    objs = {s: defaultdict(list) for s in SPLITS}
    for cat_dir in sorted(OBJ_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.png")):
            objs[assign_split(f"obj:{f.name}")][cat_dir.name].append(f)
    return pages, objs


# ---------------------------------------------------------------- compositing
def composite(page: Image.Image, obj_path: Path, rng: random.Random) -> Image.Image:
    page = page.convert("RGBA")
    W, H = page.size
    obj = Image.open(obj_path).convert("RGBA")

    # scale
    long_target = rng.uniform(*SCALE_RANGE) * max(W, H)
    s = long_target / max(obj.size)
    obj = obj.resize((max(1, int(obj.width * s)), max(1, int(obj.height * s))), Image.LANCZOS)
    # rotate (expand canvas)
    obj = obj.rotate(rng.uniform(*ROT_RANGE), expand=True, resample=Image.BICUBIC)
    # feather alpha to soften the cut edge
    r, g, b, a = obj.split()
    a = a.filter(ImageFilter.GaussianBlur(1.0))
    # occasionally make the object translucent (clear plastic/film/sleeve) so the
    # model also learns near-invisible overlays, not just opaque objects
    if rng.random() < 0.18:
        a = a.point(lambda v: int(v * rng.uniform(0.35, 0.60)))
    obj = Image.merge("RGBA", (r, g, b, a))
    ow, oh = obj.size

    # position: anywhere on the page incl. edges/corners and partial off-page
    # (real objects often sit at margins / hang off the edge)
    cx = rng.uniform(0.08, 0.92) * W
    cy = rng.uniform(0.08, 0.92) * H
    x = int(cx - ow / 2)
    y = int(cy - oh / 2)
    # keep at least ~35% of the object on the page
    x = max(-int(ow * 0.65), min(x, W - int(ow * 0.35)))
    y = max(-int(oh * 0.65), min(y, H - int(oh * 0.35)))

    # soft drop shadow from the object's alpha
    dx, dy = (rng.randint(*SHADOW_OFFSET), rng.randint(*SHADOW_OFFSET))
    sh_alpha = a.filter(ImageFilter.GaussianBlur(SHADOW_BLUR)).point(
        lambda v: int(v * rng.uniform(*SHADOW_OPACITY)))
    shadow_rgba = Image.merge("RGBA", (Image.new("L", obj.size, 0),) * 3 + (sh_alpha,))
    sh_layer = Image.new("RGBA", page.size, (0, 0, 0, 0))
    sh_layer.paste(shadow_rgba, (x + dx, y + dy), shadow_rgba)
    page = Image.alpha_composite(page, sh_layer)

    # object
    obj_layer = Image.new("RGBA", page.size, (0, 0, 0, 0))
    obj_layer.paste(obj, (x, y), obj)
    page = Image.alpha_composite(page, obj_layer)
    return page.convert("RGB")


# ---------------------------------------------------------------- preview
def preview(pages, objs, n=24):
    rng = random.Random(SEED)
    all_pages = pages["train"] + pages["val"] + pages["test"]
    cells = []
    cats = sorted({c for s in objs for c in objs[s]})
    for _ in range(n):
        page = Image.open(rng.choice(all_pages))
        cat = rng.choice(cats)
        pool = [f for s in objs for f in objs[s][cat]]
        comp = composite(page, rng.choice(pool), rng)
        comp.thumbnail((300, 460))
        cells.append((cat, comp))
    cols = 6
    cw = max(c.width for _, c in cells) + 8
    ch = max(c.height for _, c in cells) + 22
    R = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw + 8, R * ch + 8), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for i, (cat, c) in enumerate(cells):
        r, col = divmod(i, cols)
        x, y = 8 + col * cw, 8 + r * ch
        d.text((x, y), cat, fill=(150, 0, 0), font=font)
        sheet.paste(c, (x, y + 16))
    out = DATASET / "_preview_composites.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("preview ->", out)


# ---------------------------------------------------------------- full gen
# Worker functions live at module level so they are picklable for the process pool
# (Windows uses 'spawn', which re-imports this module in each worker).
def _save_good(task) -> None:
    split, i, page_path = task
    out = DATASET / split / "good" / f"{split}_good_{i:05d}.jpg"
    Image.open(page_path).convert("RGB").save(out, quality=88)


def _save_bad(task) -> None:
    split, i, page_path, obj_path, seed = task
    rng = random.Random(seed)  # per-task RNG -> deterministic and pool-safe
    comp = composite(Image.open(page_path), Path(obj_path), rng)
    cat = Path(obj_path).parent.name
    out = DATASET / split / "bad" / f"{split}_bad_{i:05d}_{cat}.jpg"
    comp.save(out, quality=88)


def generate(pages, objs, workers=None):
    """Balanced 1:1 per split: every unique page once as `good`, an equal number
    of composites as `bad`. No exact-duplicate good images; augmentation is left
    to the training loader.

    Compositing is embarrassingly parallel, so the per-image work is dispatched to
    a process pool. Page/object/seed for each `bad` image are pre-picked here with
    the master RNG, so the dataset stays fully deterministic for a given SEED
    regardless of worker count."""
    workers = workers or os.cpu_count() or 1
    rng = random.Random(SEED)
    for split in SPLITS:
        sp_pages = pages[split]
        sp_cats = [c for c in objs[split] if objs[split][c]]
        (DATASET / split / "good").mkdir(parents=True, exist_ok=True)
        (DATASET / split / "bad").mkdir(parents=True, exist_ok=True)

        good_tasks = [(split, i, str(p)) for i, p in enumerate(sp_pages)]
        bad_tasks = []
        for i in range(len(sp_pages)):
            cat = rng.choice(sp_cats)
            obj = rng.choice(objs[split][cat])
            page = rng.choice(sp_pages)
            bad_tasks.append((split, i, str(page), str(obj), rng.randrange(2**31)))

        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_save_good, good_tasks, chunksize=8))
            list(ex.map(_save_bad, bad_tasks, chunksize=8))
        print(f"  {split}: {len(sp_pages)} good + {len(sp_pages)} bad "
              f"(objcats={len(sp_cats)})")
    print(f"\nDataset -> {DATASET}  ({workers} workers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--workers", type=int, default=None,
                    help="composite worker processes (default: all CPU cores)")
    args = ap.parse_args()
    pages, objs = build_pools()
    print("split sizes (pages):", {s: len(pages[s]) for s in SPLITS})
    print("split sizes (objects):", {s: sum(len(v) for v in objs[s].values()) for s in SPLITS})
    if args.preview:
        preview(pages, objs)
    else:
        generate(pages, objs, workers=args.workers)


if __name__ == "__main__":
    main()
