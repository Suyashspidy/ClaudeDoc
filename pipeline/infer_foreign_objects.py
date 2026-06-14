"""
Inference: flag scanned book pages that have a foreign object on them.

Loads the trained EfficientNetV2-S checkpoint (weights + the recall-tuned decision
threshold) and runs it over a PDF (or a folder of PDFs). A page is flagged when
P(bad) >= threshold. Prints a summary, writes a JSON report, and optionally dumps
the flagged page images for review.

Usage:
  python infer_foreign_objects.py "C:\\path\\to\\book.pdf"
  python infer_foreign_objects.py "C:\\folder\\of\\pdfs" --out report.json --save-crops
  python infer_foreign_objects.py book.pdf --threshold 0.2   # override threshold
"""
import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF
import torch
from PIL import Image
from torchvision import models, transforms

CKPT = Path(__file__).with_name("fo_classifier_effv2s.pt")
RENDER_LONG = 1100          # render long side before resize (matches training pages)
BATCH = 16
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(device):
    ck = torch.load(CKPT, map_location=device)
    model = models.efficientnet_v2_s()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((ck["img"], ck["img"])),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    bad_idx = ck["classes"].index("bad")
    return model, tf, float(ck["threshold"]), bad_idx


def render_page(page) -> Image.Image:
    zoom = RENDER_LONG / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


@torch.no_grad()
def score_pdf(pdf_path, model, tf, bad_idx, device):
    """Return list of P(bad) per page (and keep rendered pages for optional crops)."""
    doc = fitz.open(pdf_path)
    imgs = [render_page(doc[i]) for i in range(len(doc))]
    doc.close()
    probs = []
    for i in range(0, len(imgs), BATCH):
        batch = torch.stack([tf(im) for im in imgs[i:i + BATCH]]).to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(batch)
        p = torch.softmax(logits.float(), 1)[:, bad_idx].cpu().tolist()
        probs.extend(p)
    return probs, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a PDF file or a folder containing PDFs")
    ap.add_argument("--out", default=None, help="JSON report path (default: <path>_foreign_objects.json)")
    ap.add_argument("--threshold", type=float, default=None, help="override decision threshold")
    ap.add_argument("--save-crops", action="store_true", help="save flagged page images for review")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tf, thr, bad_idx = load_model(device)
    if args.threshold is not None:
        thr = args.threshold
    print(f"device={device}  threshold={thr:.2f}")

    src = Path(args.path)
    pdfs = sorted(src.glob("*.pdf")) if src.is_dir() else [src]
    if not pdfs:
        print("no PDFs found at", src)
        return

    report = {"threshold": thr, "documents": []}
    crop_root = (src if src.is_dir() else src.parent) / "_flagged_pages"
    for pdf in pdfs:
        probs, imgs = score_pdf(pdf, model, tf, bad_idx, device)
        flagged = [{"page": i + 1, "p_object": round(p, 3)}
                   for i, p in enumerate(probs) if p >= thr]
        report["documents"].append({
            "pdf": pdf.name, "pages": len(probs), "flagged_count": len(flagged),
            "flagged_pages": [f["page"] for f in flagged], "detail": flagged,
        })
        pages_str = ", ".join(str(f["page"]) for f in flagged) or "(none)"
        print(f"\n{pdf.name}: {len(flagged)}/{len(probs)} pages flagged")
        print(f"  pages with foreign object: {pages_str}")
        if args.save_crops and flagged:
            d = crop_root / pdf.stem
            d.mkdir(parents=True, exist_ok=True)
            for f in flagged:
                imgs[f["page"] - 1].save(d / f"p{f['page']:04d}_{f['p_object']:.2f}.jpg", quality=88)

    out = Path(args.out) if args.out else (
        (src / "foreign_objects_report.json") if src.is_dir()
        else src.with_name(src.stem + "_foreign_objects.json"))
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out}")
    if args.save_crops:
        print(f"Flagged page images -> {crop_root}")


if __name__ == "__main__":
    main()
