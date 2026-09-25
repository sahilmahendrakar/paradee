"""Recompute every shard's waveform with the CPU teacher decoder from the stored intermediates,
with a fixed per-sentence seed, so the corpus is deterministic and device-canonical (CPU numerics).
Shards already carrying 'audio_cpu' are skipped; the shard is rewritten atomically."""
import sys, glob, os, time, torch
sys.path.insert(0, "scripts")
from common import load_teacher
from student import alignment
pipe, m, _ = load_teacher(device="cpu", fuse=False); ref = pipe.load_voice("af_heart")
t0 = time.time(); secs = 0
for path in sorted(glob.glob("data/teacher/shard_*.pt")):
    rows = torch.load(path)
    if rows and rows[0].get("audio_cpu"): continue
    with torch.no_grad():
        for i, r in enumerate(rows):
            torch.manual_seed(i); aln = alignment(r["pred_dur"]); Fr = aln.shape[1]
            asr = r["t_en"].float()[None] @ aln; s = ref[len(r["ids"]) - 3]
            au = m.decoder(asr, r["F0"][None, :2*Fr], r["N"][None, :2*Fr], s[:, :128]).squeeze()
            r["audio_mps"] = r["audio"]; r["audio"] = (au.clamp(-1, 1) * 32767).short(); r["audio_cpu"] = True; secs += len(au) / 24000
    torch.save(rows, path + ".tmp"); os.replace(path + ".tmp", path)
    print(f"{path} {secs/3600:.2f} h  {time.time()-t0:.0f}s  {secs/(time.time()-t0):.1f}x RT", flush=True)
print("done")
