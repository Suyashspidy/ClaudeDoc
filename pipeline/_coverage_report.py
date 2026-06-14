import csv, statistics as st
from collections import defaultdict

rows = defaultdict(list)
path = r"D:\E\Data Science Projects\ClaudeDoc\Data\_objects\manifest.csv"
with open(path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows[r["category"]].append((float(r["coverage"]), r["status"], r["pdf"], r["page"]))

print("coverage = fraction of the frame the cutout fills (high => likely whole-page grab)\n")
print(f'{"category":14s} {"n":>3} {"min":>5} {"med":>5} {"max":>5}  {"#cov>0.45":>9}')
for c in sorted(rows):
    covs = [x[0] for x in rows[c]]
    hi = sum(1 for v in covs if v > 0.45)
    print(f'{c:14s} {len(covs):3d} {min(covs):5.2f} {st.median(covs):5.2f} {max(covs):5.2f}  {hi:9d}')

print("\nSuspect (coverage > 0.45) — likely whole-page grabs, not clean objects:")
for c in sorted(rows):
    for cov, status, pdf, page in sorted(rows[c], reverse=True):
        if cov > 0.45:
            print(f'  {c:12s} cov={cov:.2f}  {pdf} p{page}')
