"""
Bulk-fetch varied public-domain SCANNED books from the Internet Archive to use as
clean BASE pages for foreign-object compositing.

Why scans (not born-digital ebooks): production input = real scanner output, so the
"good" class must span the general look of *scanned* pages — varied fonts, paper
tone, scan texture, DPI and JPEG-compression levels. Training on too few books made
the classifier learn book identity instead of object presence; this pulls a diverse
pool to fix that.

It uses the IA search API to discover candidates (biased toward single-column prose
across several decades/subjects for diversity), verifies each item is a genuine scan
(JP2/Abbyy present, not a lending-only item), downloads the PDF derivative, and skips
anything already in the output folder. Idempotent and resumable.

Usage:
  python pipeline/fetch_archive_books.py --n 50
  python pipeline/fetch_archive_books.py --n 100 --min-mb 4 --max-mb 80
  python pipeline/fetch_archive_books.py --list-only --n 30      # dry run, no download
  python pipeline/fetch_archive_books.py --query 'mediatype:texts AND subject:poetry' --n 20
"""
from __future__ import annotations
import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DEFAULT = Path(__file__).resolve().parents[1] / "Data" / "_base_src"
UA = {"User-Agent": "Mozilla/5.0 (claudedoc base-scan fetch; contact: local)"}

# Subject-varied queries so we don't get 100 lookalike books. Each is restricted to
# genuine scans of single-column-ish prose, English, public-domain era, and excludes
# lending-only / print-disabled collections (those have no public PDF).
_BASE_FILTER = (
    'mediatype:texts AND language:(English) '
    'AND NOT collection:(inlibrary) AND NOT collection:(printdisabled)'
)
_SUBJECT_QUERIES = [
    f'{_BASE_FILTER} AND subject:(fiction)',
    f'{_BASE_FILTER} AND subject:(literature)',
    f'{_BASE_FILTER} AND subject:(novel)',
    f'{_BASE_FILTER} AND subject:(biography)',
    f'{_BASE_FILTER} AND subject:(history)',
    f'{_BASE_FILTER} AND subject:(essays)',
    f'{_BASE_FILTER} AND subject:(short stories)',
    f'{_BASE_FILTER} AND subject:(poetry)',
]

# Scan-format markers in item metadata that prove it's a real library scan.
_SCAN_FORMATS = {
    "Single Page Processed JP2 ZIP", "Single Page Processed JP2 Tar",
    "Abbyy GZ", "DjVuTXT", "Scandata", "Item Tile",
}


def _get_json(url: str, timeout: int = 60, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # transient network / rate limit
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def search_identifiers(query: str, year_min: int, year_max: int, rows: int) -> list[str]:
    """Return up to `rows` identifiers matching `query` within the year range."""
    q = f"({query}) AND year:[{year_min} TO {year_max}]"
    params = {
        "q": q,
        "fl[]": "identifier",
        "rows": str(rows),
        "page": "1",
        "output": "json",
        "sort[]": "downloads desc",  # popular items are more likely cleanly scanned
    }
    url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params, doseq=True)
    data = _get_json(url)
    docs = data.get("response", {}).get("docs", [])
    return [d["identifier"] for d in docs if "identifier" in d]


def pick_pdf(meta: dict):
    """Return (filename, size_bytes, is_scan) for the best PDF derivative, or None."""
    files = meta.get("files", [])
    fmts = {f.get("format") for f in files}
    is_scan = any(k in fmts for k in _SCAN_FORMATS)
    pdfs = [f for f in files if f.get("name", "").lower().endswith(".pdf")]
    if not pdfs:
        return None
    # Prefer the 'Text PDF' derivative, then the largest.
    pdfs.sort(key=lambda f: (f.get("format") == "Text PDF", int(f.get("size", 0) or 0)),
              reverse=True)
    best = pdfs[0]
    return best["name"], int(best.get("size", 0) or 0), is_scan


def download(url: str, dest: Path, timeout: int = 900) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    tmp.replace(dest)
    return total


def build_pool(queries: list[str], year_min: int, year_max: int, per_query: int, seed: int) -> list[str]:
    """Collect identifiers from every query, dedupe, and shuffle for diversity."""
    seen, pool = set(), []
    for q in queries:
        try:
            ids = search_identifiers(q, year_min, year_max, per_query)
        except Exception as e:
            print(f"  ! search failed for query, skipping: {e}")
            continue
        for ident in ids:
            if ident not in seen:
                seen.add(ident)
                pool.append(ident)
    random.Random(seed).shuffle(pool)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch varied public-domain scanned books from the Internet Archive.")
    ap.add_argument("--n", type=int, default=50, help="number of NEW books to download (default 50)")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output folder (default Data/_base_src)")
    ap.add_argument("--query", default=None, help="override the discovery query (otherwise uses varied subjects)")
    ap.add_argument("--year-min", type=int, default=1850)
    ap.add_argument("--year-max", type=int, default=1965)
    ap.add_argument("--min-mb", type=float, default=3.0, help="skip PDFs smaller than this (likely not a real scan)")
    ap.add_argument("--max-mb", type=float, default=100.0, help="skip PDFs larger than this (slow / image-heavy)")
    ap.add_argument("--require-scan", action="store_true", default=True, help="only accept verified library scans")
    ap.add_argument("--allow-nonscan", dest="require_scan", action="store_false", help="accept non-scan PDFs too")
    ap.add_argument("--pool-per-query", type=int, default=60, help="candidates to pull per subject query")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed for reproducibility")
    ap.add_argument("--list-only", action="store_true", help="dry run: print what would be downloaded")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in out.glob("*.pdf")}
    print(f"Output: {out}  ({len(existing)} books already present)")

    queries = [args.query] if args.query else _SUBJECT_QUERIES
    pool = build_pool(queries, args.year_min, args.year_max, args.pool_per_query, args.seed)
    print(f"Discovered {len(pool)} candidate identifiers; want {args.n} new download(s).\n")

    got = 0
    for ident in pool:
        if got >= args.n:
            break
        if ident in existing:
            continue
        dest = out / f"{ident}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            continue

        try:
            meta = _get_json(f"https://archive.org/metadata/{ident}")
        except Exception as e:
            print(f"[skip] {ident}: metadata failed ({e})")
            continue

        picked = pick_pdf(meta)
        if not picked:
            print(f"[skip] {ident}: no PDF derivative")
            continue
        name, size, is_scan = picked
        size_mb = size / 1e6

        if args.require_scan and not is_scan:
            print(f"[skip] {ident}: not a verified scan")
            continue
        if size_mb and not (args.min_mb <= size_mb <= args.max_mb):
            print(f"[skip] {ident}: size {size_mb:.1f} MB outside [{args.min_mb}, {args.max_mb}]")
            continue

        if args.list_only:
            print(f"[would get] {ident}  scan={is_scan}  ~{size_mb:.1f} MB  ({name})")
            got += 1
            continue

        url = f"https://archive.org/download/{ident}/{urllib.parse.quote(name)}"
        print(f"[get {got + 1}/{args.n}] {ident}  ~{size_mb:.1f} MB ...", end="", flush=True)
        try:
            total = download(url, dest)
            got += 1
            print(f" done ({total / 1e6:.1f} MB)")
        except Exception as e:
            print(f" FAILED ({e})")
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
        time.sleep(0.6)  # be polite to archive.org

    action = "would download" if args.list_only else "downloaded"
    print(f"\nDone. {action} {got} book(s) -> {out}")
    if not args.list_only and got < args.n:
        print("Tip: re-run to fetch more (it resumes), widen --year-min/--year-max, "
              "raise --pool-per-query, or pass a custom --query.")


if __name__ == "__main__":
    main()
