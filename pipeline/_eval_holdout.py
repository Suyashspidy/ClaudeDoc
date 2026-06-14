"""Run the saved model on the real phone-photo holdout; list which pages were MISSED."""
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, models, transforms

ROOT = Path(r"D:\E\Data Science Projects\ClaudeDoc\Data")
HOLD = ROOT / "_holdout_real"
CKPT = Path(r"D:\E\Data Science Projects\ClaudeDoc\pipeline\fo_classifier_effv2s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location=device)
IMG, thr = ck["img"], ck["threshold"]

tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds = datasets.ImageFolder(HOLD, transform=tf)
model = models.efficientnet_v2_s()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(ck["state_dict"]); model.to(device).eval()

print(f"threshold={thr:.2f}  classes={ds.classes}  n={len(ds)}")
missed, probs = [], []
with torch.no_grad():
    for i, (x, y) in enumerate(ds):
        p_bad = torch.softmax(model(x.unsqueeze(0).to(device)).float(), 1)[0, 0].item()
        probs.append(p_bad)
        if p_bad < thr:  # predicted good = MISSED a real bad page
            missed.append((Path(ds.samples[i][0]).name, p_bad))

probs = np.array(probs)
print(f"caught {(probs>=thr).mean()*100:.1f}%  (mean p_bad={probs.mean():.2f})")
print(f"\nMISSED {len(missed)} real bad pages (predicted good):")
for name, p in sorted(missed, key=lambda t: t[1]):
    print(f"  p_bad={p:.2f}  {name}")
