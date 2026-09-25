"""UTMOS (automatic naturalness MOS, 1-5) for WAV files or directories. usage: utmos.py PATH [PATH ...]"""
import sys, glob, os, warnings, torch, soundfile as sf, torchaudio, numpy as np
warnings.filterwarnings("ignore")
model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
def score(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    with torch.no_grad(): return model(torch.tensor(x)[None], sr).item()
if __name__ == "__main__":
    files = []
    for p in sys.argv[1:]: files += sorted(glob.glob(f"{p}/*.wav")) if os.path.isdir(p) else [p]
    groups = {}
    for f in files:
        s = score(f); key = os.path.basename(f).split("_", 2)[-1].replace(".wav", "") if "held_" in f else os.path.basename(f)
        groups.setdefault(os.path.dirname(f) + " :: " + key, []).append(s)
    for k, v in groups.items(): print(f"{k:60s} UTMOS {np.mean(v):.2f} (n={len(v)})")
