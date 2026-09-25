"""Buzz experiment 1, no training: lock the student's phase in 2-8 kHz to its own harmonic source.
In voiced frames, each STFT bin's phase becomes source phase + a time-smoothed version of (student phase -
source phase), keeping the student's magnitude. L = smoothing length in frames (hop 256), or "seg" for one
offset per voiced run. Listener (2026-09-23): 9 frames slightly better; 17 frames as good as teacher phase.
usage: phase_lock.py [CKPT] [N_SENT] [--full]   (--full: student text side too, i.e. the whole Paradee pipeline)
writes out/phase_lock/ or out/phase_lock_full/"""
import sys, glob, os, warnings, torch, torch.nn.functional as F, soundfile as sf; warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from student import alignment
from student_decoder import load_student_decoder
n, hop, SR = 1024, 256, 24000; W = torch.hann_window(n); FR = torch.arange(n // 2 + 1) * SR / n
BAND = ((FR >= 2000) & (FR < 8000))[:, None]

def smooth(Z, L, v):
    if L == "seg":  # mean over each voiced run
        out = torch.zeros_like(Z); j = 0; vv = v[0].tolist()
        while j < len(vv):
            if not vv[j]: j += 1; continue
            e = j
            while e < len(vv) and vv[e]: e += 1
            out[:, j:e] = Z[:, j:e].mean(1, keepdim=True); j = e
        return out
    k = torch.hann_window(L + 2)[1:-1]; k = (k / k.sum())[None, None]
    return torch.complex(F.conv1d(Z.real[:, None], k, padding=L // 2)[:, 0], F.conv1d(Z.imag[:, None], k, padding=L // 2)[:, 0])

def lock(s, src, f0, L):
    """s: student waveform, src: its harmonic excitation, f0: F0 curve at 300-sample hops."""
    k = min(len(s), len(src)); s, src = s[:k], src[:k]
    S = torch.stft(s, n, hop, window=W, return_complex=True); E = torch.stft(src, n, hop, window=W, return_complex=True)
    v = torch.tensor([float(f0[min(j * hop // 300, len(f0) - 1)]) > 60 for j in range(S.shape[1])])[None]
    rel = S * E.conj() / E.abs().clamp_min(1e-6)
    ph = torch.where(BAND & v, E.angle() + smooth(rel, L, v).angle(), S.angle())
    return torch.istft(torch.polar(S.abs(), ph), n, hop, window=W, length=k)

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; FULL = "--full" in sys.argv
    ck = args[0] if args else "checkpoints/dec_A_gan3/last.pt"; NS = int(args[1]) if len(args) > 1 else 2
    held = torch.load(sorted(glob.glob("data/teacher/shard_*.pt"))[0])[:NS]
    ds = load_student_decoder("A", ck); cap = {}
    ds.generator.m_source.register_forward_hook(lambda m, i, o: cap.__setitem__("src", o[0].squeeze()))
    if FULL:
        from student import TextStudent
        ts = TextStudent("s+mlp").eval(); ts.load_state_dict(torch.load("checkpoints/text_s+mlp/last.pt", map_location="cpu")["model"])
    tag = "_" + os.path.basename(os.path.dirname(ck))
    out = ("out/phase_lock_full" if FULL else "out/phase_lock") + tag; os.makedirs(out, exist_ok=True)
    for i, r in enumerate(held):
        with torch.no_grad():
            if FULL:
                ids = r["ids"][None]; Ln = torch.tensor([len(r["ids"])])
                pd, _, _, pasr = ts(ids, Ln); aln = alignment(torch.round(pd[0]).clamp(min=1).long())[None]
                _, F0, N, _ = ts(ids, Ln, aln); asr = pasr @ aln
            else:
                aln = alignment(r["pred_dur"]); asr = (r["t_en"].float() @ aln)[None]; Fr = aln.shape[1]; F0 = r["F0"][None, :2 * Fr]; N = r["N"][None, :2 * Fr]
            torch.manual_seed(i); s = ds(asr, F0, N).squeeze()
        sf.write(f"{out}/held_{i + 1}_student.wav", s.numpy(), SR)
        for L in (9, 17, 33, "seg"):
            sf.write(f"{out}/held_{i + 1}_locked{L}.wav", lock(s, cap["src"], F0[0], L).numpy(), SR)
    print("written", out)
