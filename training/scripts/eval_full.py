"""Full student pipeline: text student -> student decoder, no teacher at inference. Reports DTW log-mel
vs the teacher waveform on held-out sentences, alongside each half alone, and saves listening WAVs.
usage: eval_full.py TEXT_PRESET DEC_PRESET [--device mps] [--n 8]"""
import sys, glob, argparse, warnings, torch, soundfile as sf, os
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from student import TextStudent, alignment
from student_decoder import StudentDecoder
from common import load_teacher, compare, cpu_source_stft
ap = argparse.ArgumentParser(); ap.add_argument("text"); ap.add_argument("dec"); ap.add_argument("--device", default="mps"); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--text-tag", default=""); ap.add_argument("--dec-tag", default="")
A = ap.parse_args(); DEV = A.device
held = torch.load(sorted(glob.glob("data/teacher/shard_*.pt"))[0])[:A.n]
ts = TextStudent(A.text).to(DEV).eval(); ts.load_state_dict(torch.load(f"checkpoints/text_{A.text}{A.text_tag}/last.pt", map_location=DEV)["model"])
from student_decoder import load_student_decoder
ds = cpu_source_stft(load_student_decoder(A.dec, f"checkpoints/dec_{A.dec}{A.dec_tag}/last.pt", device=DEV))
pipe, teacher, _ = load_teacher(device=DEV, fuse=False); ref = pipe.load_voice("af_heart"); cpu_source_stft(teacher.decoder)
out_dir = f"out/full_{A.text}{A.text_tag}_{A.dec}{A.dec_tag}"; os.makedirs(out_dir, exist_ok=True)
@torch.no_grad()
def text_outputs(r):
    ids = r["ids"][None].to(DEV); L = torch.tensor([len(r["ids"])], device=DEV)
    pd, _, _, pasr = ts(ids, L); pred_dur = torch.round(pd[0]).clamp(min=1).long(); aln = alignment(pred_dur)[None]
    _, pF0, pN, _ = ts(ids, L, aln); return pasr @ aln, pF0, pN
@torch.no_grad()
def teacher_inputs(r):
    aln = alignment(r["pred_dur"]).to(DEV); Fr = aln.shape[1]
    return r["t_en"].float()[None].to(DEV) @ aln, r["F0"][None, :2 * Fr].to(DEV), r["N"][None, :2 * Fr].to(DEV)
refs = [r["audio"].float() / 32767 for r in held]
rows = {"teacher text + teacher decoder (floor)": [], "student text + teacher decoder": [], "teacher text + student decoder": [], "student text + student decoder (full)": []}
with torch.no_grad():
    for i, r in enumerate(held):
        s = ref[len(r["ids"]) - 3].to(DEV)[:, :128]
        ta, tf, tn = teacher_inputs(r); sa, sf_, sn = text_outputs(r)
        rows["teacher text + teacher decoder (floor)"].append(teacher.decoder(ta, tf, tn, s).squeeze().cpu())
        rows["student text + teacher decoder"].append(teacher.decoder(sa, sf_, sn, s).squeeze().cpu())
        rows["teacher text + student decoder"].append(ds(ta, tf, tn).squeeze().cpu())
        full = ds(sa, sf_, sn).squeeze().cpu(); rows["student text + student decoder (full)"].append(full)
        if i < 3: sf.write(f"{out_dir}/held_{i+1}_full_student.wav", full.numpy(), 24000)
print(f"text {A.text} ({sum(p.numel() for p in ts.parameters())/1e6:.2f}M) + decoder {A.dec} ({sum(p.numel() for p in ds.parameters())/1e6:.2f}M), {A.n} held-out sentences")
print(f"{'pipeline':44s} {'dtwL1':>6s} {'|dur|ms':>8s}")
for k, v in rows.items():
    raw, dtw, ld = compare(refs, v); print(f"{k:44s} {dtw:6.3f} {ld:8.0f}")
