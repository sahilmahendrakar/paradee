"""Export the whole Paradee pipeline (text student + stage-two decoder + locked33 phase filter) as ONE
ONNX graph for the browser: input_ids [1,T] int64 (with the 0 pad token at both ends), speed [1] float
-> waveform [1,S] float at 24 kHz. The phase filter is rewritten with conv1d DFT bases (band bins only)
so it needs no complex ops, then checked against scripts/phase_lock.py on the same waveform.
Also writes a weight-only int8 copy (int8 weights + DequantizeLinear), which is what the extension ships.
usage: export_paradee.py [--text checkpoints/text_s+mlp/last.pt] [--dec checkpoints/dec_A_gan3/last.pt] [--out models/paradee]"""
import sys, glob, os, time, argparse, warnings, torch, numpy as np, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from student import TextStudent
from student_decoder import load_student_decoder
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
import phase_lock

ap = argparse.ArgumentParser(); ap.add_argument("--text", default="checkpoints/text_s+mlp/last.pt"); ap.add_argument("--dec", default="checkpoints/dec_A_gan3/last.pt")
ap.add_argument("--out", default="models/paradee"); ap.add_argument("--L", type=int, default=33)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
N_FFT, HOP, SR = 1024, 256, 24000
BINS = torch.nonzero(phase_lock.BAND[:, 0]).squeeze(1)          # 2-8 kHz: bins 86..341

def strip(m):
    for x in m.modules():
        try: remove_weight_norm(x)
        except Exception: pass
        if is_parametrized(x, "weight"):
            try: remove_parametrizations(x, "weight", leave_parametrized=True)
            except Exception: pass

class Lock(nn.Module):
    """phase_lock.lock() without complex numbers. Output = s + iSTFT(change inside the band), so bins outside
    2-8 kHz and unvoiced frames pass through untouched."""
    def __init__(self, L):
        super().__init__()
        w = torch.hann_window(N_FFT); n = torch.arange(N_FFT).float(); k = BINS.float()[:, None]
        ang = 2 * torch.pi * k * n / N_FFT
        self.register_buffer("fwd", torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w])[:, None])      # [2K,1,N]
        self.register_buffer("inv", (torch.cat([torch.cos(ang), -torch.sin(ang)]) * (2 / N_FFT) * w)[:, None])  # [2K,1,N] (band bins have weight 2)
        self.register_buffer("w2", (w ** 2)[None, None])
        sk = torch.hann_window(L + 2)[1:-1]; self.register_buffer("sk", (sk / sk.sum())[None, None]); self.pad = L // 2
    def stft(self, x):   # x [1,S] -> re, im [K, frames], reflect-padded like torch.stft(center=True)
        y = F.conv1d(F.pad(x[:, None], (N_FFT // 2, N_FFT // 2), mode="reflect"), self.fwd, stride=HOP)[0]
        return y[:len(BINS)], y[len(BINS):]
    def forward(self, s, src, f0):
        k = torch.minimum(torch.tensor(s.shape[1]), torch.tensor(src.shape[1])); s = s[:, :k]; src = src[:, :k]
        Sr, Si = self.stft(s); Er, Ei = self.stft(src)
        nf = Sr.shape[1]; j = torch.clamp(torch.arange(nf) * HOP // 300, max=f0.shape[1] - 1)
        v = (f0[0, j] > 60).float()[None]                                                   # voiced frames
        Em = torch.sqrt(Er ** 2 + Ei ** 2).clamp_min(1e-6)
        Rr, Ri = (Sr * Er + Si * Ei) / Em, (Si * Er - Sr * Ei) / Em                          # S * conj(E) / |E|
        sm = lambda z: F.conv1d(z[:, None], self.sk, padding=self.pad)[:, 0]
        Rr, Ri = sm(Rr), sm(Ri); Rm = torch.sqrt(Rr ** 2 + Ri ** 2).clamp_min(1e-12)
        Sm = torch.sqrt(Sr ** 2 + Si ** 2)
        ur, ui = (Er * Rr - Ei * Ri) / (Em * Rm), (Er * Ri + Ei * Rr) / (Em * Rm)            # unit(E) * unit(smoothed)
        Dr, Di = (Sm * ur - Sr) * v, (Sm * ui - Si) * v
        ola = F.conv_transpose1d(torch.cat([Dr, Di])[None], self.inv, stride=HOP)[0]       # inverse DFT, window, overlap-add
        env = F.conv_transpose1d(torch.ones(1, 1, nf), self.w2, stride=HOP)[0]
        d = (ola / env.clamp_min(1e-8))[:, N_FFT // 2: N_FFT // 2 + k]
        return s + d

class ConvSTFT(nn.Module):
    """Stand-in for kokoro's CustomSTFT (the ONNX-friendly one) that matches torch.stft / torch.istft, which the
    decoder was trained with. CustomSTFT costs about 0.5 UTMOS here, for three reasons: replicate instead of reflect
    padding, float noise in the DC and Nyquist imaginary parts whose random sign flips the phase feature between +pi
    and -pi (~2.5% of strong bins), and an inverse that neither halves DC/Nyquist nor divides by the window overlap."""
    def __init__(self, n, hop):
        super().__init__(); self.n, self.hop = n, hop; w = torch.hann_window(n); k = torch.arange(n // 2 + 1).float()[:, None]
        ang = 2 * torch.pi * k * torch.arange(n).float() / n
        re, im = torch.cos(ang) * w, -torch.sin(ang) * w; im[0] = 0; im[-1] = 0
        self.register_buffer("fwd", torch.cat([re, im])[:, None])
        c = torch.full((n // 2 + 1, 1), 2.0); c[0] = c[-1] = 1
        self.register_buffer("inv", (torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w)[:, None])
        self.register_buffer("w2", (w ** 2)[None, None])
    def transform(self, x):
        y = F.conv1d(F.pad(x[:, None], (self.n // 2, self.n // 2), mode="reflect"), self.fwd, stride=self.hop); K = self.n // 2 + 1
        re, im = y[:, :K], y[:, K:]
        ph = torch.where((im == 0) & (re < 0), torch.full_like(re, torch.pi), torch.atan2(im, re))   # torch.angle gives +pi there
        return torch.sqrt(re ** 2 + im ** 2), ph
    def inverse(self, mag, phase):
        z = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], 1)
        y = F.conv_transpose1d(z, self.inv, stride=self.hop)[:, 0]
        env = F.conv_transpose1d(torch.ones(1, 1, mag.shape[-1]), self.w2, stride=self.hop)[:, 0]
        return (y / env.clamp_min(1e-8))[:, self.n // 2: -(self.n // 2)][:, None]

class Paradee(nn.Module):
    def __init__(self, ts, ds, L):
        super().__init__(); self.ts, self.ds, self.lock = ts, ds, Lock(L); self.cap = {}
        ds.generator.m_source.register_forward_hook(lambda m, i, o: self.cap.__setitem__("src", o[0]))
    def text(self, ids):
        """TextStudent.forward for one unpadded sentence, without masks or packed sequences."""
        t = self.ts; p = t.predictor; s = t.style
        d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
        x = d_en; T = ids.shape[1]; sx = s[:, :, None].expand(-1, -1, T)
        x = torch.cat([x, sx], 1)
        for b in p.text_encoder.lstms:
            if isinstance(b, nn.LSTM): x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
            else: x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
        d = x.transpose(-1, -2)                                            # [1,T,hd+sd]
        dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]      # [T]
        e = t.text_encoder; h = e.embedding(ids).transpose(1, 2)
        for c in e.cnn: h = c(h)
        t_en = e.lstm(h.transpose(1, 2))[0]
        asr_tok = t.asr_proj(t_en).transpose(-1, -2)                       # [1,512,T]
        return d, dur, asr_tok
    def forward(self, input_ids, speed):
        d, dur, asr_tok = self.text(input_ids)
        n = torch.clamp(torch.round(dur / speed), min=1).long()
        end = torch.cumsum(n, 0); start = end - n; f = torch.arange(end[-1])[None]
        aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]   # [1,T,F]
        F0, N = self.ts.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, self.ts.style)
        s = self.ds(asr_tok @ aln, F0, N).squeeze(1)
        return self.lock(s, self.cap["src"].squeeze(-1), F0)

def load(disable_complex=True):
    ts = TextStudent("s+mlp").eval(); ts.load_state_dict(torch.load(A.text, map_location="cpu")["model"])
    ds = load_student_decoder("A", A.dec, disable_complex=disable_complex); strip(ts); strip(ds)
    if disable_complex: ds.generator.stft = ConvSTFT(ds.generator.stft.filter_length, ds.generator.stft.hop_length)
    return Paradee(ts, ds, A.L).eval()

held = torch.load(sorted(glob.glob("data/teacher/shard_*.pt"))[0])[:4]

# 1. the rewritten filter matches phase_lock.lock() on the same student waveform and source
pm = load(disable_complex=False)
with torch.no_grad():
    ids = held[0]["ids"][None]; torch.manual_seed(0); _ = pm(ids, torch.ones(1))
    d, dur, asr_tok = pm.text(ids); n = torch.clamp(torch.round(dur), min=1).long()
    from student import alignment; aln = alignment(n)[None]
    F0, N = pm.ts.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, pm.ts.style)
    torch.manual_seed(0); s = pm.ds(asr_tok @ aln, F0, N).squeeze(); src = pm.cap["src"].squeeze()
    ref = phase_lock.lock(s, src, F0[0], A.L); mine = pm.lock(s[None], src[None], F0)[0]
print(f"filter check: max |mine - phase_lock| = {(mine - ref).abs().max():.2e} (signal peak {ref.abs().max():.2f}), "
      f"filter changed the signal by rms {(ref - s[:len(ref)]).pow(2).mean().sqrt():.4f}")

# 2. export
m = load(disable_complex=True)
path = f"{A.out}/paradee.onnx"
ids = held[1]["ids"][None]; torch.manual_seed(0)
torch.onnx.export(m, (ids, torch.ones(1)), path, input_names=["input_ids", "speed"], output_names=["waveform"],
                  dynamic_axes={"input_ids": {1: "tokens"}, "waveform": {1: "samples"}}, opset_version=17, dynamo=False)
print(f"exported {path}: {os.path.getsize(path)/1e6:.1f} MB")

# 3. small copy (what the extension ships, and what the paper's 8.45 MB refers to): every weight matrix in int8 per
# output channel (int8 + DequantizeLinear), other float vectors in fp16 (+ Cast), and the phase filter's DFT bases
# (4 MB in fp32) rebuilt at load time from a few ops instead of stored
import onnx
from onnx import numpy_helper, helper, TensorProto
g = onnx.load(path); nq = nh = 0
fwd = torch.cat([torch.cos(2 * torch.pi * BINS.double()[:, None] * torch.arange(N_FFT).double() / N_FFT),
                 -torch.sin(2 * torch.pi * BINS.double()[:, None] * torch.arange(N_FFT).double() / N_FFT)]) * torch.hann_window(N_FFT, dtype=torch.float64)
def basis_nodes(name, scale):
    """[2K,1,N] = cat(cos, -sin)(2 pi k n / N) * hann(n) * scale, with k*n reduced mod N in int64 so fp32 cos stays exact"""
    p = name + "_gen_"; K = len(BINS)
    g.graph.initializer.extend([numpy_helper.from_array(BINS.numpy().astype(np.int64)[:, None], p + "k"), numpy_helper.from_array(np.arange(N_FFT, dtype=np.int64)[None], p + "n"),
        numpy_helper.from_array(np.array(N_FFT, np.int64), p + "N"), numpy_helper.from_array(np.array(2 * np.pi / N_FFT, np.float32), p + "c"),
        numpy_helper.from_array((torch.hann_window(N_FFT) * scale).numpy().astype(np.float32), p + "w")])
    return [helper.make_node("Mul", [p + "k", p + "n"], [p + "kn"]), helper.make_node("Mod", [p + "kn", p + "N"], [p + "m"]),
            helper.make_node("Cast", [p + "m"], [p + "mf"], to=TensorProto.FLOAT), helper.make_node("Mul", [p + "mf", p + "c"], [p + "a"]),
            helper.make_node("Cos", [p + "a"], [p + "cos"]), helper.make_node("Sin", [p + "a"], [p + "sin"]), helper.make_node("Neg", [p + "sin"], [p + "nsin"]),
            helper.make_node("Concat", [p + "cos", p + "nsin"], [p + "cs"], axis=0), helper.make_node("Mul", [p + "cs", p + "w"], [p + "b"]),
            helper.make_node("Unsqueeze", [p + "b", p + "ax"], [name])], numpy_helper.from_array(np.array([1], np.int64), p + "ax")
new_nodes = []
for init in list(g.graph.initializer):
    w = numpy_helper.to_array(init); name = init.name
    if w.dtype != np.float32: continue
    if w.shape == (2 * len(BINS), 1, N_FFT):                                          # a DFT basis: forward (scale 1) or inverse (2/N)
        scale = float(np.abs(w).max() / fwd.abs().max()); scale = 1.0 if abs(scale - 1) < 1e-3 else 2 / N_FFT
        assert np.abs(w - fwd[:, None].numpy() * scale).max() < 1e-3 * scale, name   # the stored fp32 bases carry ~1e-4 error from cos of angles up to 2000 rad; the rebuilt ones are exact
        g.graph.initializer.remove(init); nodes, ax = basis_nodes(name, scale); g.graph.initializer.append(ax); new_nodes += nodes
    elif w.ndim >= 2 and w.size >= 1024:
        sc = np.abs(w).reshape(w.shape[0], -1).max(1) / 127; sc[sc == 0] = 1
        q = np.clip(np.round(w / sc.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int8)
        g.graph.initializer.remove(init)
        g.graph.initializer.extend([numpy_helper.from_array(q, name + "_q"), numpy_helper.from_array(sc.astype(np.float16), name + "_s16"),
                                    numpy_helper.from_array(np.zeros(len(sc), np.int8), name + "_z")])
        new_nodes += [helper.make_node("Cast", [name + "_s16"], [name + "_s"], to=TensorProto.FLOAT),
                      helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name], axis=0)]; nq += 1
    elif w.size >= 64 and np.abs(w).max() < 6e4:                                      # biases, norms: fp16 (scalars stay fp32, e.g. the 1e-8 clamps)
        g.graph.initializer.remove(init); g.graph.initializer.append(numpy_helper.from_array(w.astype(np.float16), name + "_16"))
        new_nodes.append(helper.make_node("Cast", [name + "_16"], [name], to=TensorProto.FLOAT)); nh += 1
for n in reversed(new_nodes): g.graph.node.insert(0, n)
qpath = f"{A.out}/paradee_int8.onnx"; onnx.checker.check_model(g); onnx.save(g, qpath)
print(f"int8 weights for {nq} tensors, fp16 for {nh}, DFT bases generated -> {qpath}: {os.path.getsize(qpath)/1e6:.2f} MB")

# 4. ORT vs torch, speed on one thread
import onnxruntime as ort, soundfile as sf
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
for p in (path, qpath):
    sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]); tot = secs = 0
    for i, r in enumerate(held):
        feed = {"input_ids": r["ids"][None].numpy(), "speed": np.ones(1, np.float32)}
        t = time.perf_counter(); y = sess.run(None, feed)[0]; tot += time.perf_counter() - t; secs += y.shape[1] / SR
        sf.write(f"{A.out}/held_{i + 1}_{os.path.basename(p)[:-5]}.wav", y[0], SR)
    with torch.no_grad(): torch.manual_seed(0); yt = m(held[0]["ids"][None], torch.ones(1))
    print(f"{os.path.basename(p)}: length vs torch {sess.run(None, {'input_ids': held[0]['ids'][None].numpy(), 'speed': np.ones(1, np.float32)})[0].shape[1]} vs {yt.shape[1]}, "
          f"{secs:.1f}s audio in {tot:.2f}s -> {secs / tot:.1f}x realtime on 1 thread")
