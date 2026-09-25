"""Stage A1: train a small text-side student against teacher intermediates, evaluate through the
frozen teacher decoder.  usage: train_text.py PRESET [--steps N] [--eval-only]
Resumable from checkpoints/text_PRESET/last.pt."""
import sys, os, glob, math, time, json, argparse, warnings, random
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F, numpy as np, soundfile as sf
sys.path.insert(0, "scripts")
from student import TextStudent, alignment
ap = argparse.ArgumentParser(); ap.add_argument("preset"); ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--device", default="mps")
ap.add_argument("--shards", type=int, default=999); ap.add_argument("--tag", default="")
ap.add_argument("--through-decoder", action="store_true", help="add a log-mel loss through the frozen teacher decoder on a random segment")
ap.add_argument("--td-bs", type=int, default=8); ap.add_argument("--seg", type=int, default=48); ap.add_argument("--td-weight", type=float, default=1.0)
ap.add_argument("--asr-weight", type=float, default=1.0)
ap.add_argument("--through-student-decoder", default="", help="checkpoint of a StudentDecoder to use as the frozen decoder (preset from --dec-preset) instead of the teacher decoder")
ap.add_argument("--dec-preset", default="A")
A = ap.parse_args(); DEV = A.device
RUN = f"checkpoints/text_{A.preset}{A.tag}"; os.makedirs(RUN, exist_ok=True)

def load_rows():
    """Held-out = first 200 rows of shard 0 (fixed forever); train = everything else that exists."""
    shards = sorted(glob.glob("data/teacher/shard_*.pt"))[:A.shards]
    def lean(r, train):  # drop what training never reads; keeps RAM at ~1/4 of the shard size
        r.pop("audio_mps", None); r.pop("d", None)
        if train and not A.through_decoder: r.pop("audio", None)
        return r
    rows0 = torch.load(shards[0])
    held = [lean(r, False) for r in rows0[:200]]
    train = [lean(r, True) for r in rows0[200:]] + [lean(r, True) for sh in shards[1:] for r in torch.load(sh)]
    return held, train, len(shards)
held, train, nsh = load_rows()
print(f"{len(train)} train / {len(held)} held-out sentences from {nsh} shards", flush=True)

def collate(batch):
    B = len(batch); T = max(len(r["ids"]) for r in batch); Fm = max(len(r["F0"]) for r in batch) // 2
    ids = torch.zeros(B, T, dtype=torch.long); L = torch.tensor([len(r["ids"]) for r in batch])
    dur = torch.zeros(B, T); ten = torch.zeros(B, 512, T); aln = torch.zeros(B, T, Fm)
    F0 = torch.zeros(B, 2 * Fm); N = torch.zeros(B, 2 * Fm); fm = torch.zeros(B, 2 * Fm, dtype=torch.bool)
    for i, r in enumerate(batch):
        t = len(r["ids"]); f = len(r["F0"]) // 2
        ids[i, :t] = r["ids"]; dur[i, :t] = r["dur"]; ten[i, :, :t] = r["t_en"].float()
        aln[i, :t, :f] = alignment(r["pred_dur"], f); F0[i, :2 * f] = r["F0"]; N[i, :2 * f] = r["N"]; fm[i, :2 * f] = True
    tm = torch.arange(T)[None] < L[:, None]
    nb = min(A.td_bs, B) if A.through_decoder else 0; win = torch.zeros(nb, dtype=torch.long); aud = torch.zeros(nb, 600 * A.seg); ok = torch.zeros(nb, dtype=torch.bool)
    for i in range(nb):
        f = len(batch[i]["F0"]) // 2
        if f >= A.seg:
            a = random.randint(0, f - A.seg); win[i] = a; aud[i] = batch[i]["audio"][600 * a:600 * (a + A.seg)].float() / 32767; ok[i] = True
    return ids, L, dur, ten, aln, F0, N, tm, fm, win, aud, ok

def losses(model, batch):
    ids, L, dur, ten, aln, F0, N, tm, fm, win, aud, ok = [x.to(DEV) for x in batch]
    pd, pF0, pN, pasr = model(ids, L, aln)
    l_dur = (F.l1_loss(torch.log(pd.clamp_min(1e-3)), torch.log(dur.clamp_min(1e-3)), reduction="none") * tm).sum() / tm.sum()
    l_f0 = (F.l1_loss(pF0, F0, reduction="none") * fm).sum() / fm.sum() / 100
    l_n = (F.l1_loss(pN, N, reduction="none") * fm).sum() / fm.sum()
    l_asr = (F.l1_loss(pasr, ten, reduction="none") * tm[:, None]).sum() / tm.sum() / 512 * A.asr_weight
    if not A.through_decoder: return l_dur, l_f0, l_n, l_asr
    l_td = torch.zeros((), device=DEV); n = 0
    for i in range(len(win)):
        if not ok[i]: continue
        a = int(win[i]); sl = aln[i][:, a:a + A.seg]
        asr_seg = (pasr[i] @ sl)[None]; F0_seg = pF0[i:i + 1, 2 * a:2 * (a + A.seg)]; N_seg = pN[i:i + 1, 2 * a:2 * (a + A.seg)]
        sty = TD_REF[int(L[i]) - 3].to(DEV)[:, :128]
        y = TD_DEC(asr_seg, F0_seg, N_seg, sty).squeeze(1)
        k = min(y.shape[-1], aud.shape[-1])
        l_td = l_td + F.l1_loss(torch.log(TD_MEL(y[:, :k]).clamp_min(1e-5)), torch.log(TD_MEL(aud[i:i + 1, :k]).clamp_min(1e-5))); n += 1
    return l_dur, l_f0, l_n, l_asr, l_td / max(n, 1) * A.td_weight

model = TextStudent(A.preset).to(DEV)
if A.through_decoder:
    import torchaudio
    from common import load_teacher, cpu_source_stft
    _pipe, _teacher, _ = load_teacher(device=DEV, fuse=False); TD_REF = _pipe.load_voice("af_heart")
    if A.through_student_decoder:
        from student_decoder import StudentDecoder
        _sd = StudentDecoder(A.dec_preset).to(DEV); _sd.load_state_dict(torch.load(A.through_student_decoder, map_location=DEV)["model"])
        _sdec = cpu_source_stft(_sd).eval().requires_grad_(False)
        TD_DEC = lambda asr, f0, n, sty: _sdec(asr, f0, n); print("through-decoder loss uses student decoder", A.through_student_decoder, flush=True)
    else:
        TD_DEC = cpu_source_stft(_teacher.decoder).eval().requires_grad_(False)
    del _teacher
    TD_MEL = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80).to(DEV)
nparam = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01)
sched = lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, A.steps) / A.steps))
step = 0; last = f"{RUN}/last.pt"
if os.path.exists(last):
    ck = torch.load(last, map_location=DEV); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    print(f"resumed at step {step}", flush=True)

@torch.no_grad()
def evaluate(n_audio=8, save=False):
    """Teacher-forced losses on held-out, plus free-running synthesis through the frozen teacher decoder."""
    model.eval(); acc = None; nb = 0
    for i in range(0, len(held), 32):
        ls = np.array([l.item() for l in losses(model, collate(held[i:i + 32]))]); acc = ls if acc is None else acc + ls; nb += 1
    acc /= nb
    from common import load_teacher, compare, cpu_source_stft
    pipe, teacher, _ = load_teacher(device=DEV, fuse=False); ref = pipe.load_voice("af_heart"); cpu_source_stft(teacher.decoder)
    outs, refs, dmae, f0r = [], [], [], []
    for r in held[:n_audio]:
        ids = r["ids"][None].to(DEV); L = torch.tensor([len(r["ids"])], device=DEV)
        pd, _, _, pasr = model(ids, L)
        pred_dur = torch.round(pd[0]).clamp(min=1).long(); aln = alignment(pred_dur)[None]
        pd2, pF0, pN, _ = model(ids, L, aln)
        asr = pasr @ aln; s = ref[len(r["ids"]) - 3].to(DEV)
        audio = teacher.decoder(asr, pF0, pN, s[:, :128]).squeeze().cpu()
        outs.append(audio); refs.append(r["audio"].float() / 32767)
        dmae.append((pred_dur.cpu() - r["pred_dur"]).abs().float().mean().item())
        f = min(len(pF0[0]), len(r["F0"])); v = r["F0"][:f] > 30
        f0r.append(((pF0[0, :f].cpu() - r["F0"][:f])[v] ** 2).mean().sqrt().item())
        if save: sf.write(f"{RUN}/held_{len(outs)}_student.wav", audio.numpy(), 24000); sf.write(f"{RUN}/held_{len(outs)}_teacher.wav", refs[-1].numpy(), 24000)
    raw, dtw, ld = compare(refs, outs)
    model.train()
    return dict(step=step, l_dur=acc[0], l_f0=acc[1], l_n=acc[2], l_asr=acc[3], l_td=(acc[4] if len(acc) > 4 else None), dur_mae_frames=float(np.mean(dmae)),
                f0_rmse_hz=float(np.mean(f0r)), dtw_l1=dtw, dur_delta_ms=ld)

if A.eval_only:
    print(json.dumps(evaluate(save=True), indent=1)); sys.exit()
print(f"student {A.preset}: {nparam/1e6:.2f}M params, device {DEV}", flush=True)
model.train(); t0 = time.time(); log = open(f"{RUN}/log.jsonl", "a")
while step < A.steps:
    batch = collate(random.sample(train, A.bs))
    for g in opt.param_groups: g["lr"] = A.lr * sched(step)
    ls = losses(model, batch); loss = sum(ls)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
    if step % 50 == 0:
        print(f"step {step} loss {loss.item():.4f} dur {ls[0].item():.4f} f0 {ls[1].item():.4f} n {ls[2].item():.4f} asr {ls[3].item():.4f}" + (f" td {ls[4].item():.4f}" if len(ls) > 4 else "") + f" {(time.time()-t0)/step:.2f}s/step", flush=True)
    if step % 500 == 0 or step == A.steps:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
        if len(glob.glob("data/teacher/shard_*.pt")[:A.shards]) > nsh:
            held, train, nsh = load_rows(); print(f"reloaded: {len(train)} train sentences from {nsh} shards", flush=True)
    if step % 2000 == 0 or step == A.steps:
        ev = evaluate(save=(step == A.steps)); print("EVAL", json.dumps(ev), flush=True); log.write(json.dumps(ev) + "\n"); log.flush()
print("done")
