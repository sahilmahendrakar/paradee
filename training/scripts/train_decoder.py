"""Stage A2: train a small acoustic-side student (StudentDecoder) on teacher intermediates
(asr, F0, N) to reproduce the teacher waveform. Losses: log-mel L1 + multi-resolution STFT.
usage: train_decoder.py PRESET [--steps N]   resumable from checkpoints/dec_PRESET/last.pt"""
import sys, os, glob, math, time, json, argparse, warnings, random
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F, numpy as np, soundfile as sf, torchaudio
sys.path.insert(0, "scripts")
from student_decoder import StudentDecoder
from student import alignment
from common import cpu_source_stft
ap = argparse.ArgumentParser(); ap.add_argument("preset"); ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--seg", type=int, default=64)
ap.add_argument("--gan", action="store_true", help="add MPD+MRD adversarial and feature-matching losses")
ap.add_argument("--gan-weight", type=float, default=1.0); ap.add_argument("--fm-weight", type=float, default=2.0); ap.add_argument("--mel-weight", type=float, default=45.0)
ap.add_argument("--init-from", default="", help="checkpoint to initialise the generator from (fresh optimiser)")
ap.add_argument("--d-mult", type=int, default=1, help="discriminator channel multiplier"); ap.add_argument("--d-lr-mult", type=float, default=1.0)
ap.add_argument("--matched-source", action="store_true", help="use the teacher's excitation (copied mixing layer + per-sentence seed) so waveform-level losses are valid")
ap.add_argument("--wave-weight", type=float, default=0.0); ap.add_argument("--cstft-weight", type=float, default=0.0)
ap.add_argument("--head-weight", type=float, default=0.0, help="distil the teacher iSTFT head: L1 on the log-magnitude branch (needs --matched-source)")
ap.add_argument("--phase-weight", type=float, default=0.0, help="L1 on the head phase branch sin(x) (needs --matched-source)")
ap.add_argument("--harmonics", type=int, default=9, help="sine harmonics in the student excitation (teacher: 9); the generator runs the source under no_grad, so the mixing layer never trains: new columns are fixed at -0.2, the mean magnitude of the teacher's 9")
ap.add_argument("--hf-weight", type=float, default=0.0, help="band-limited (1.5-8 kHz) STFT loss, window 1024 hop 128: the band where mel cannot resolve harmonics")
ap.add_argument("--src-bands", action="store_true", help="feed every harmonic's phase as its own (cos, sin) channel pair to each generator stage (zero-init)")
ap.add_argument("--full-phase", action="store_true", help="add an unbounded, zero-init phase term to the head's sin(x) phase (which spans only +-1 rad)")
ap.add_argument("--utmos", action="store_true", help="score held-out audio with UTMOS at every EVAL")
ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--device", default="mps"); ap.add_argument("--profile", action="store_true"); ap.add_argument("--shards", type=int, default=999); ap.add_argument("--tag", default="")
A = ap.parse_args(); DEV = A.device; RUN = f"checkpoints/dec_{A.preset}{A.tag}"; os.makedirs(RUN, exist_ok=True)
HEAD = A.head_weight > 0 or A.phase_weight > 0
if HEAD: assert A.matched_source, "--head-weight/--phase-weight need --matched-source"

def load_rows():
    shards = sorted(glob.glob("data/teacher/shard_*.pt"))[:A.shards]
    def lean(r):
        for k in ("audio_mps", "d", "dur", "ids"): r.pop(k, None)
        return r
    def load(sh):
        rows = torch.load(sh)
        for i, r in enumerate(rows): r["seed"] = i
        return [lean(r) for r in rows]
    rows0 = load(shards[0])
    return rows0[:200], rows0[200:] + [r for sh in shards[1:] for r in load(sh)], len(shards)
held, train, nsh = load_rows(); print(f"{len(train)} train / {len(held)} held-out from {nsh} shards", flush=True)

def full(r):
    aln = alignment(r["pred_dur"]); asr = r["t_en"].float() @ aln; Fr = aln.shape[1]
    return asr, r["F0"][:2 * Fr], r["N"][:2 * Fr], r["audio"][:600 * Fr].float() / 32767
def segment(r, seg):
    asr, F0, N, au = full(r); Fr = asr.shape[1]
    if Fr <= seg:  # pad short
        pad = seg - Fr; asr = F.pad(asr, (0, pad)); F0 = F.pad(F0, (0, 2 * pad)); N = F.pad(N, (0, 2 * pad)); au = F.pad(au, (0, 600 * pad)); a = 0
    else: a = random.randint(0, Fr - seg)
    src = matched_source(r, a, seg) if A.matched_source else torch.zeros(1)
    if A.matched_source and len(src) < 600 * seg: src = F.pad(src, (0, 600 * seg - len(src)))
    return asr[:, a:a + seg], F0[2 * a:2 * (a + seg)], N[2 * a:2 * (a + seg)], au[600 * a:600 * (a + seg)], src, style(r)
def style(r): return REF[len(r["ps"]) - 1][0, :128] if HEAD else torch.zeros(1)   # teacher's per-length af_heart style (decoder half)
def collate(rows, seg):
    x = [segment(r, seg) for r in rows]
    return [torch.stack([t[i] for t in x]).to(DEV) for i in range(6)]
def matched_source(r, a, seg):
    """Teacher's harmonic source for this sentence (same seed as regen_audio.py), cropped to the segment."""
    F0 = r["F0"][None]
    torch.manual_seed(r["seed"])
    with torch.no_grad(): har = SRC_M(SRC_UP(F0[:, None]).transpose(1, 2))[0]   # [1, samples, 1]
    return har[0, 600 * a:600 * (a + seg), 0]
class FixedSource(torch.nn.Module):
    def __init__(self): super().__init__(); self.src = None
    def forward(self, f0): return self.src, None, None

mels = [torchaudio.transforms.MelSpectrogram(24000, n_fft=n, hop_length=n // 4, n_mels=80).to(DEV) for n in (1024,)]
stfts = [(n, n // 4) for n in (512, 1024, 2048)]
def spec_losses(y, t):
    lm = sum(F.l1_loss(torch.log(m(y).clamp_min(1e-5)), torch.log(m(t).clamp_min(1e-5))) for m in mels)
    ls = 0
    for n, h in stfts:
        w = torch.hann_window(n, device=DEV)
        Y = torch.stft(y, n, h, window=w, return_complex=True).abs(); T = torch.stft(t, n, h, window=w, return_complex=True).abs()
        ls = ls + (T - Y).norm() / T.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Y.clamp_min(1e-5)), torch.log(T.clamp_min(1e-5)))
    return lm, ls / len(stfts)

model = StudentDecoder(A.preset, harmonics=A.harmonics, bands=A.src_bands, full_phase=A.full_phase).to(DEV); cpu_source_stft(model)
if A.matched_source:
    from common import load_teacher
    import copy
    _p, _t, _ = load_teacher(device="cpu", fuse=False); SRC_M = copy.deepcopy(_t.decoder.generator.m_source); SRC_UP = _t.decoder.generator.f0_upsamp
    model.generator.m_source.load_state_dict(SRC_M.state_dict())
    for q in model.generator.m_source.parameters(): q.requires_grad_(False)
    FIXED = FixedSource(); model.generator.m_source = FIXED
    if HEAD:  # frozen teacher decoder on the same excitation; hooks capture both conv_post outputs (pre exp/sin)
        REF = _p.load_voice("af_heart"); TEACHER_DEC = _t.decoder.to(DEV).eval().requires_grad_(False); cpu_source_stft(TEACHER_DEC)
        TEACHER_DEC.generator.m_source = FIXED; CAP = {}
        TEACHER_DEC.generator.conv_post.register_forward_hook(lambda m, i, o: CAP.__setitem__("t", o))
        model.generator.conv_post.register_forward_hook(lambda m, i, o: CAP.__setitem__("s", o))
    del _t
nparam = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, betas=(0.8, 0.99), weight_decay=0.01)
sched = lambda s: min(1, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, A.steps) / A.steps))
def widen(sd0):
    """The generator runs the source under no_grad, so the mixing layer never trains: use the checkpoint's (else the teacher's) 9 weights plus fixed -0.2 columns."""
    if A.harmonics == 9: return sd0
    if "generator.m_source.l_linear.weight" not in sd0:
        from common import load_teacher
        _t = load_teacher(device="cpu", fuse=False)[1]; sd0.update({"generator.m_source." + k: v.to(DEV) for k, v in _t.decoder.generator.m_source.state_dict().items()}); del _t
    w9 = sd0["generator.m_source.l_linear.weight"]; sd0["generator.m_source.l_linear.weight"] = F.pad(w9, (0, A.harmonics - w9.shape[1]), value=-0.2); return sd0
step = 0; last = f"{RUN}/last.pt"
if A.gan:
    from discriminators import Discriminators, d_loss, g_loss
    D = Discriminators(mult=A.d_mult).to(DEV); optD = torch.optim.AdamW(D.parameters(), lr=A.lr * A.d_lr_mult, betas=(0.8, 0.99), weight_decay=0.01)
if os.path.exists(last):
    ck = torch.load(last, map_location=DEV); model.load_state_dict(ck["model"], strict=not A.matched_source); opt.load_state_dict(ck["opt"]); step = ck["step"]; print(f"resumed at {step}", flush=True)
    if A.gan and "D" in ck: D.load_state_dict(ck["D"]); optD.load_state_dict(ck["optD"])
elif not A.init_from and A.harmonics != 9: model.load_state_dict(widen({}), strict=False)
elif A.init_from:
    sd0 = widen(torch.load(A.init_from, map_location=DEV)["model"])
    NEW = A.src_bands or A.full_phase; res = model.load_state_dict(sd0, strict=not (A.matched_source or NEW))
    if NEW: assert all("band_convs" in k or "conv_phase" in k for k in res.missing_keys) and not res.unexpected_keys, res; print(f"new zero-init params: {res.missing_keys}", flush=True)
    print(f"initialised generator from {A.init_from}", flush=True)

@torch.no_grad()
def evaluate(n_audio=8, save=False):
    model.eval(); from common import compare
    outs, refs, lm_acc = [], [], []
    for r in held[:n_audio]:
        asr, F0, N, au = full(r)
        if A.matched_source: FIXED.src = matched_source(r, 0, asr.shape[1])[None, :, None].to(DEV)
        y = model(asr[None].to(DEV), F0[None].to(DEV), N[None].to(DEV)).squeeze().cpu()
        k = min(len(y), len(au)); outs.append(y[:k]); refs.append(au[:k])
        if save or A.utmos: sf.write(f"{RUN}/held_{len(outs)}_student.wav", y.numpy(), 24000); sf.write(f"{RUN}/held_{len(outs)}_teacher.wav", au.numpy(), 24000)
    raw, dtw, _ = compare(refs, outs); model.train()
    if A.utmos:
        from utmos import score; u = sum(score(f"{RUN}/held_{i + 1}_student.wav") for i in range(len(outs))) / len(outs)
        os.makedirs(f"{RUN}/eval_{step}", exist_ok=True); [os.replace(f"{RUN}/held_{i + 1}_student.wav", f"{RUN}/eval_{step}/held_{i + 1}_student.wav") for i in range(len(outs))]
        return {"step": step, "dtw_l1": dtw, "raw_l1": raw, "utmos": round(float(u), 3)}
    return dict(step=step, dtw_l1=dtw, raw_l1=raw)

if A.eval_only: print(json.dumps(evaluate(save=True))); sys.exit()
print(f"decoder {A.preset}: {nparam/1e6:.2f}M params, device {DEV}", flush=True)
model.train(); t0 = time.time(); log = open(f"{RUN}/log.jsonl", "a")
PT = {}
def tick(name, t0):
    if not A.profile: return t0
    if DEV == "mps": torch.mps.synchronize()
    PT[name] = PT.get(name, 0) + time.perf_counter() - t0; return time.perf_counter()
while step < A.steps:
    t1 = time.perf_counter()
    asr, F0, N, au, src, sty = collate(random.sample(train, A.bs), A.seg); t1 = tick("collate", t1)
    if A.matched_source: FIXED.src = src[:, :, None]
    for g in opt.param_groups: g["lr"] = A.lr * sched(step)
    y = model(asr, F0, N).squeeze(1)
    k = min(y.shape[-1], au.shape[-1]); y, au = y[:, :k], au[:, :k]; lm, ls = spec_losses(y, au); t1 = tick("G fwd+spec", t1)
    if A.gan:
        ld = d_loss(D, au, y); optD.zero_grad(); ld.backward(); optD.step(); t1 = tick("D step", t1)
        adv, fm = g_loss(D, au, y); loss = A.mel_weight * (lm + ls) + A.gan_weight * adv + A.fm_weight * fm; t1 = tick("G adv", t1)
    else: loss = lm + ls
    if A.matched_source and (A.wave_weight > 0 or A.cstft_weight > 0):
        w1 = torch.hann_window(1024, device=DEV); Y = torch.stft(y, 1024, 256, window=w1, return_complex=True); T_ = torch.stft(au, 1024, 256, window=w1, return_complex=True)
        l_wave = F.l1_loss(y, au); l_cstft = (Y - T_).abs().mean() / T_.abs().mean().clamp_min(1e-5)
        loss = loss + A.wave_weight * l_wave + A.cstft_weight * l_cstft
    if A.hf_weight > 0:
        w1 = torch.hann_window(1024, device=DEV); Yh = torch.stft(y, 1024, 128, window=w1, return_complex=True).abs()[:, 64:342]; Th = torch.stft(au, 1024, 128, window=w1, return_complex=True).abs()[:, 64:342]
        l_hf = (Th - Yh).norm() / Th.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Yh.clamp_min(1e-5)), torch.log(Th.clamp_min(1e-5))); loss = loss + A.hf_weight * l_hf
    if HEAD:
        with torch.no_grad(): TEACHER_DEC(asr, F0, N, sty)
        ht, hs = CAP["t"], CAP["s"]; k2 = min(ht.shape[-1], hs.shape[-1]); ht, hs = ht[..., :k2], hs[..., :k2]
        l_mag = F.l1_loss(hs[:, :11], ht[:, :11]); l_ph = F.l1_loss(torch.sin(hs[:, 11:]), torch.sin(ht[:, 11:]))
        loss = loss + A.head_weight * l_mag + A.phase_weight * l_ph; t1 = tick("head", t1)
    opt.zero_grad(); loss.backward(); t1 = tick("G bwd", t1); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); step += 1; t1 = tick("clip+opt", t1)
    if A.profile and step % 50 == 0: print("PROFILE " + "  ".join(f"{k} {v/50*1000:.0f}ms" for k, v in PT.items()), flush=True); PT.clear()
    if step % 50 == 0: print(f"step {step} loss {loss.item():.4f} mel {lm.item():.4f} stft {ls.item():.4f}" + (f" adv {adv.item():.3f} fm {fm.item():.3f} d {ld.item():.3f}" if A.gan else "") + (f" wave {l_wave.item():.4f} cstft {l_cstft.item():.4f}" if A.matched_source and (A.wave_weight > 0 or A.cstft_weight > 0) else "") + (f" hf {l_hf.item():.4f}" if A.hf_weight > 0 else "") + (f" hmag {l_mag.item():.4f} hph {l_ph.item():.4f}" if HEAD else "") + f" {(time.time()-t0)/step:.2f}s/step", flush=True)
    if step % 500 == 0 or step == A.steps:
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
        if A.gan: ck["D"] = D.state_dict(); ck["optD"] = optD.state_dict()
        torch.save(ck, last)
        if len(glob.glob("data/teacher/shard_*.pt")[:A.shards]) > nsh: held, train, nsh = load_rows(); print(f"reloaded {len(train)} from {nsh} shards", flush=True)
    if step % 2500 == 0 or step == A.steps:
        ev = evaluate(save=(step == A.steps)); print("EVAL", json.dumps(ev), flush=True); log.write(json.dumps(ev) + "\n"); log.flush()
print("done")
