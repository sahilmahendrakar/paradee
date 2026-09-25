"""Whisper (base) word error rate of WAVs against the text in a .txt file next to each one.
usage: wer.py DIR [DIR ...]"""
import sys, re, glob, warnings; warnings.filterwarnings("ignore")
import soundfile as sf, torch, torchaudio, whisper, jiwer
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())
m = whisper.load_model("base")
for d in sys.argv[1:]:
    R, H = [], []
    for f in sorted(glob.glob(f"{d}/*.wav")):
        x, sr = sf.read(f, dtype="float32"); x = torchaudio.functional.resample(torch.tensor(x), sr, 16000).numpy()
        H.append(norm(m.transcribe(x, language="en", fp16=False)["text"])); R.append(norm(open(f[:-4] + ".txt").read()))
    print(f"{d}: WER {jiwer.wer(R, H)*100:.1f}% over {len(R)} sentences")
