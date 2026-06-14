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
import random
from collections import defaultdict
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
        for pno in range(len(doc)):
            img = render_page(doc[pno])
            if not (INK_MIN < ink_fraction(img) < INK_MAX):
                skipped += 1
                continue
            out = PAGE_CACHE / f"{pdf.stem}__p{pno:04d}.jpg"
            img.save(out, quality=90)
            kept.append(out)
        doc.close()
        print(f"  cached {pdf.stem}: running total {len(kept)} kept")
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


def build_pools():
    pages = defaultdict(list)
    for p in build_page_cache():
        pages[assign_split(p.name)].append(p)
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
def generate(pages, objs):
    """Balanced 1:1 per split: every unique page once as `good`, an equal number
    of composites as `bad`. No exact-duplicate good images; augmentation is left
    to the training loader."""
    rng = random.Random(SEED)
    for split in SPLITS:
        sp_pages = pages[split]
        sp_cats = [c for c in objs[split] if objs[split][c]]
        good_dir = DATASET / split / "good"
        bad_dir = DATASET / split / "bad"
        good_dir.mkdir(parents=True, exist_ok=True)
        bad_dir.mkdir(parents=True, exist_ok=True)
        # good: each unique page saved once
        for i, p in enumerate(sp_pages):
            Image.open(p).convert("RGB").save(
                good_dir / f"{split}_good_{i:05d}.jpg", quality=88)
        # bad: equal count, random page x random (stratified) object
        for i in range(len(sp_pages)):
            cat = rng.choice(sp_cats)
            obj = rng.choice(objs[split][cat])
            comp = composite(Image.open(rng.choice(sp_pages)), obj, rng)
            comp.save(bad_dir / f"{split}_bad_{i:05d}_{cat}.jpg", quality=88)
        print(f"  {split}: {len(sp_pages)} good + {len(sp_pages)} bad "
              f"(objcats={len(sp_cats)})")
    print(f"\nDataset -> {DATASET}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    pages, objs = build_pools()
    print("split sizes (pages):", {s: len(pages[s]) for s in SPLITS})
    print("split sizes (objects):", {s: sum(len(v) for v in objs[s].values()) for s in SPLITS})
    if args.preview:
        preview(pages, objs)
    else:
        generate(pages, objs)


if __name__ == "__main__":
    main()
