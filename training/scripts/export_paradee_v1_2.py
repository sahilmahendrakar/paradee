"""Export Paradee 1.2 (four voices) as ONE ONNX graph: input_ids [1,T] int64, speed [1] float, voice [1] int64 (row of
the style tables, VOICES order) -> waveform [1,S] float at 24 kHz. Same graph as export_paradee_v1_1.py, with the voice
picking the text side's and the decoder's style vectors. The head is rewritten without complex numbers (conv1d DFT bases), checked against the torch head on the
same noise, then written as fp32 and as the small copy (int8 weights, fp16 vectors, DFT bases generated at load time).
usage: export_paradee_v1_2.py DEC_CKPT --text TEXT_CKPT [--config config.json] [--out models/paradee_v1.2]
With the released weights from https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2 (see training/README.md, step 6):
  export_paradee_v1_2.py hf/pytorch/decoder.pt --text hf/pytorch/text_side.pt --config hf/config.json"""
import sys, glob, os, time, argparse, warnings, torch, numpy as np, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from student import TextStudent, alignment
import json
from vhead import VDecoder, SR, FHOP
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
ap = argparse.ArgumentParser(); ap.add_argument("dec"); ap.add_argument("--text", required=True); ap.add_argument("--out", default="models/paradee_v1.2")
ap.add_argument("--config", help="config.json with the phoneme vocab (default: the one next to the folder that holds --text)")
ap.add_argument("--voices", default="af_heart,bf_isabella,am_puck,bm_fable", help="style-table order used in training")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); VOICES = A.voices.split(",")

def strip(m):
    for x in m.modules():
        try: remove_weight_norm(x)
        except Exception: pass
        if is_parametrized(x, "weight"):
            try: remove_parametrizations(x, "weight", leave_parametrized=True)
            except Exception: pass

class ConvSTFT(nn.Module):
    """torch.stft / torch.istft (hann, centre, reflect) as conv1d / conv_transpose1d with real and imaginary parts kept apart."""
    def __init__(self, n, hop):
        super().__init__(); self.n, self.hop, self.K = n, hop, n // 2 + 1; w = torch.hann_window(n, dtype=torch.float64); k = torch.arange(self.K, dtype=torch.float64)[:, None]
        ang = 2 * torch.pi * ((k * torch.arange(n, dtype=torch.float64)) % n) / n; c = torch.full((self.K, 1), 2.0, dtype=torch.float64); c[0] = c[-1] = 1
        self.register_buffer("fwd", (torch.cat([torch.cos(ang), -torch.sin(ang)]) * w).float()[:, None])
        self.register_buffer("inv", (torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w).float()[:, None]); self.register_buffer("w2", (w ** 2).float()[None, None])
    def transform(self, x):   # [B,S] -> re, im [B,K,frames]
        y = F.conv1d(F.pad(x[:, None], (self.n // 2, self.n // 2), mode="reflect"), self.fwd, stride=self.hop); return y[:, :self.K], y[:, self.K:]
    def inverse(self, re, im):
        y = F.conv_transpose1d(torch.cat([re, im], 1), self.inv, stride=self.hop)[:, 0]
        env = F.conv_transpose1d(torch.ones_like(re[:1, :1]), self.w2, stride=self.hop)[:, 0]
        return (y / env.clamp_min(1e-8))[:, self.n // 2: -(self.n // 2)]

class HeadX(nn.Module):
    """SourceFilterHead2.forward without complex ops, float64 on-device tricks or repeat_interleave."""
    def __init__(self, h):
        super().__init__(); self.h = h; self.st = ConvSTFT(h.n_fft, h.hop); self.sb = ConvSTFT(150, 75)
        self.register_buffer("k", torch.arange(1, h.K + 1).float()[None, :, None]); self.register_buffer("fr", torch.linspace(0, SR / 2, h.nb)[None, :, None])
    def forward(self, x, f0, noise=None):
        h = self.h; n = x.shape[2]; T = n * h.up; S = n * FHOP
        z = h.norm0(h.inp(x).transpose(1, 2)).transpose(1, 2)
        for b in h.blocks: z = b(z)
        hn = h.norm1(z.transpose(1, 2)); o = h.out(hn).view(1, T, h.C * h.G + 1); o = torch.cat([o, o[:, -1:]], 1)
        g = (o[..., :-1].reshape(1, T + 1, h.C, h.G) @ h.M.T).permute(0, 2, 3, 1)
        gh, gn = torch.exp(g[:, 0].clamp(max=6)), torch.exp(g[:, 3].clamp(max=6)); um = torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2).clamp_min(1e-4); ur, ui = g[:, 1] / um, g[:, 2] / um
        # sine bank (vhead.sine_bank): running phase in float64 at frame rate, wrapped, then interpolated between frame centres
        r = f0.double() * FHOP / SR; c = torch.cumsum(r, 1); c = (c - torch.floor(c)).float(); d = torch.cat([r[:, 1:], torch.zeros_like(r[:, :1])], 1).float()
        t = torch.arange(S, device=x.device); p = (t.float() + 0.5) / FHOP - 0.5; i = torch.clamp(torch.floor(p), min=0).long(); i = torch.minimum(i, (n - 1) * torch.ones_like(i)); frac = torch.clamp(p - i.float(), 0, 1)
        cyc = c[:, i] + frac * d[:, i]; f0s = f0[:, t // FHOP]; uv = (f0s > 10).float()
        amp = ((f0s[:, None] * self.k < 11000).float() * uv[:, None])
        e = (torch.sin(2 * torch.pi * self.k * cyc[:, None]) * amp).sum(1) * 0.1
        nz = torch.randn_like(e) if noise is None else noise
        Er, Ei = self.st.transform(e); Nr, Ni = self.st.transform(nz); j = 0
        for m in range(1, h.mod + 1):
            for fn in (torch.cos, torch.sin):
                a = g[:, 4 + j]; mr, mi = self.st.transform(nz * uv * fn(2 * torch.pi * m * cyc)); Nr = Nr + a * mr; Ni = Ni + a * mi; j += 1
        f0f = torch.cat([f0[:, torch.arange(T, device=x.device) // h.up], f0[:, -1:]], 1).clamp_min(60); ang = 2 * torch.pi * (self.fr / f0f[:, None]) * (0.5 * torch.tanh(o[..., -1])[:, None])
        Hr = gh * (ur * torch.cos(ang) - ui * torch.sin(ang)); Hi = gh * (ur * torch.sin(ang) + ui * torch.cos(ang))
        y = self.st.inverse(Er * Hr - Ei * Hi + Nr * gn, Er * Hi + Ei * Hr + Ni * gn)
        if h.burst:
            gb = torch.exp((h.out_b(hn).view(1, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2); gb = torch.cat([gb, gb[..., -1:]], -1)
            br, bi = self.sb.transform(nz.flip(-1)); y = y + self.sb.inverse(br * gb, bi * gb)
        return y

class Paradee2(nn.Module):
    def __init__(self, ts, vd): super().__init__(); self.ts, self.vd, self.head = ts, vd, HeadX(vd.head)
    def text(self, ids, s):
        t = self.ts; p = t.predictor
        d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
        x = d_en; T = ids.shape[1]; sx = s[:, :, None].expand(-1, -1, T); x = torch.cat([x, sx], 1)
        for b in p.text_encoder.lstms:
            if isinstance(b, nn.LSTM): x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
            else: x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
        d = x.transpose(-1, -2); dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]
        e = t.text_encoder; h = e.embedding(ids).transpose(1, 2)
        for c in e.cnn: h = c(h)
        return d, dur, t.asr_proj(e.lstm(h.transpose(1, 2))[0]).transpose(-1, -2)
    def trunk(self, asr, F0_curve, N, s):
        v = self.vd; F0 = v.F0_conv(F0_curve.unsqueeze(1)); Nn = v.N_conv(N.unsqueeze(1))
        x = v.encode(torch.cat([asr, F0, Nn], 1), s); asr_res = v.asr_res(asr); res = True
        for block in v.decode:
            if res: x = torch.cat([x, asr_res, F0, Nn], 1)
            x = block(x, s)
            if block.upsample_type != "none": res = False
        return x
    def forward(self, input_ids, speed, voice, noise=None):
        d, dur, asr_tok = self.text(input_ids, self.ts.style[voice])
        n = torch.clamp(torch.round(dur / speed), min=1).long()
        end = torch.cumsum(n, 0); start = end - n; f = torch.arange(end[-1])[None]
        aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
        F0, N = self.ts.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, self.ts.style[voice])
        return self.head(self.trunk(asr_tok @ aln, F0, N, self.vd.style[voice]), F0, noise)

ts = TextStudent("s+mlp", n_voices=len(VOICES)).eval(); ts.load_state_dict(torch.load(A.text, map_location="cpu")["model"])
vd = VDecoder(head="v2", dim=160, blocks=4, n_voices=len(VOICES)).eval(); vd.load_state_dict(torch.load(A.dec, map_location="cpu")["model"])
ts0 = TextStudent("s+mlp", n_voices=len(VOICES)).eval(); ts0.load_state_dict(ts.state_dict())   # unstripped copies for the check
vd0 = VDecoder(head="v2", dim=160, blocks=4, n_voices=len(VOICES)).eval(); vd0.load_state_dict(vd.state_dict())
strip(ts); strip(vd); m = Paradee2(ts, vd).eval()
vocab = json.load(open(A.config or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(A.text))), "config.json")))["vocab"]
held = [json.loads(l) for l in open("data/heldout_200.jsonl")][:4]   # the first held-out sentences, already in phonemes
for r in held: r["ids"] = torch.LongTensor([0, *[vocab[c] for c in r["phonemes"] if c in vocab], 0])

# 1. the export graph matches the training code path (v12_compare.paradee) for every voice on the same noise
with torch.no_grad():
    for vi, v in enumerate(VOICES):
        ids = held[0]["ids"][None]; L = torch.tensor([ids.shape[1]]); vo = torch.tensor([vi])
        pd, _, _, pasr = ts0(ids, L, voice=vo); aln = alignment(torch.round(pd[0]).clamp(min=1).long())[None]
        _, pF0, pN, _ = ts0(ids, L, aln, vo); nz = torch.randn(1, pF0.shape[1] * FHOP)
        ref = vd0(pasr @ aln, pF0, pN, noise=nz, voice=vo)[0]; mine = m(ids, torch.ones(1), vo, noise=nz)[0]
        print(f"{v}: export vs training code, max abs diff {(mine - ref).abs().max():.2e} (signal peak {ref.abs().max():.2f}, lengths {mine.shape[0]} {ref.shape[0]})")

# 2. export
path = f"{A.out}/paradee.onnx"; ids = held[1]["ids"][None]
torch.onnx.export(m, (ids, torch.ones(1), torch.zeros(1, dtype=torch.long)), path, input_names=["input_ids", "speed", "voice"], output_names=["waveform"],
                  dynamic_axes={"input_ids": {1: "tokens"}, "waveform": {1: "samples"}}, opset_version=17, dynamo=False)
print(f"exported {path}: {os.path.getsize(path)/1e6:.1f} MB")

# 3. small copy: weight matrices int8 per output channel, other float vectors fp16, DFT bases generated from a few ops
import onnx
from onnx import numpy_helper, helper, TensorProto
g = onnx.load(path); nq = nh = nb = 0; new_nodes = []
def basis(n, inverse):
    w = torch.hann_window(n, dtype=torch.float64); k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]; ang = 2 * torch.pi * ((k * torch.arange(n, dtype=torch.float64)) % n) / n
    c = torch.full((n // 2 + 1, 1), 2.0, dtype=torch.float64); c[0] = c[-1] = 1
    return ((torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w) if inverse else (torch.cat([torch.cos(ang), -torch.sin(ang)]) * w)).float()[:, None].numpy()
def basis_nodes(name, n, inverse):
    """[2K,1,n] = cat(cos, -sin)(2 pi k t / n) * per-row scale * hann(t), with k*t reduced mod n in int64 so fp32 cos stays exact"""
    p = name + "_gen_"; K = n // 2 + 1; row = np.ones((2 * K, 1), np.float32)
    if inverse: row[:] = 2 / n; row[[0, K - 1, K, 2 * K - 1]] = 1 / n
    g.graph.initializer.extend([numpy_helper.from_array(np.arange(K, dtype=np.int64)[:, None], p + "k"), numpy_helper.from_array(np.arange(n, dtype=np.int64)[None], p + "n"),
        numpy_helper.from_array(np.array(n, np.int64), p + "N"), numpy_helper.from_array(np.array(2 * np.pi / n, np.float32), p + "c"),
        numpy_helper.from_array(torch.hann_window(n).numpy()[None], p + "w"), numpy_helper.from_array(row, p + "row"), numpy_helper.from_array(np.array([1], np.int64), p + "ax")])
    return [helper.make_node("Mul", [p + "k", p + "n"], [p + "kn"]), helper.make_node("Mod", [p + "kn", p + "N"], [p + "m"]),
            helper.make_node("Cast", [p + "m"], [p + "mf"], to=TensorProto.FLOAT), helper.make_node("Mul", [p + "mf", p + "c"], [p + "a"]),
            helper.make_node("Cos", [p + "a"], [p + "cos"]), helper.make_node("Sin", [p + "a"], [p + "sin"]), helper.make_node("Neg", [p + "sin"], [p + "nsin"]),
            helper.make_node("Concat", [p + "cos", p + "nsin"], [p + "cs"], axis=0), helper.make_node("Mul", [p + "cs", p + "w"], [p + "b0"]), helper.make_node("Mul", [p + "b0", p + "row"], [p + "b"]),
            helper.make_node("Unsqueeze", [p + "b", p + "ax"], [name])]
for init in list(g.graph.initializer):
    w = numpy_helper.to_array(init); name = init.name
    if w.dtype != np.float32: continue
    hit = [(n, inv) for n in (600, 150) for inv in (False, True) if w.shape == (n + 2, 1, n) and np.abs(w - basis(n, inv)).max() < 1e-4 * np.abs(w).max()]
    if hit: g.graph.initializer.remove(init); new_nodes += basis_nodes(name, *hit[0]); nb += 1
    elif w.ndim >= 2 and w.size >= 1024:
        sc = np.abs(w).reshape(w.shape[0], -1).max(1) / 127; sc[sc == 0] = 1
        q = np.clip(np.round(w / sc.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int8); g.graph.initializer.remove(init)
        g.graph.initializer.extend([numpy_helper.from_array(q, name + "_q"), numpy_helper.from_array(sc.astype(np.float16), name + "_s16"), numpy_helper.from_array(np.zeros(len(sc), np.int8), name + "_z")])
        new_nodes += [helper.make_node("Cast", [name + "_s16"], [name + "_s"], to=TensorProto.FLOAT), helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name], axis=0)]; nq += 1
    elif w.size >= 64 and np.abs(w).max() < 6e4:
        g.graph.initializer.remove(init); g.graph.initializer.append(numpy_helper.from_array(w.astype(np.float16), name + "_16"))
        new_nodes.append(helper.make_node("Cast", [name + "_16"], [name], to=TensorProto.FLOAT)); nh += 1
for n_ in reversed(new_nodes): g.graph.node.insert(0, n_)
qpath = f"{A.out}/paradee_int8.onnx"; onnx.checker.check_model(g); onnx.save(g, qpath)
print(f"int8 weights for {nq} tensors, fp16 for {nh}, {nb} DFT bases generated -> {qpath}: {os.path.getsize(qpath)/1e6:.2f} MB")

# 4. ORT: matches torch on fixed lengths, speed on one thread, listening files per voice
import onnxruntime as ort, soundfile as sf
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
for p in (path, qpath):
    sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]); tot = secs = 0
    for vi, v in enumerate(VOICES):
        for rep in range(2):
            for i, r in enumerate(held):
                feed = {"input_ids": r["ids"][None].numpy(), "speed": np.ones(1, np.float32), "voice": np.array([vi], np.int64)}
                t = time.perf_counter(); y = sess.run(None, feed)[0]; dt = time.perf_counter() - t
                if rep: tot += dt; secs += y.shape[1] / SR; sf.write(f"{A.out}/held_{i + 1}_{v}_{os.path.basename(p)[:-5]}.wav", y[0], SR)
        with torch.no_grad(): yt = m(held[0]["ids"][None], torch.ones(1), torch.tensor([vi]))
        print(f"{os.path.basename(p)} {v}: length {sess.run(None, {'input_ids': held[0]['ids'][None].numpy(), 'speed': np.ones(1, np.float32), 'voice': np.array([vi], np.int64)})[0].shape[1]} vs torch {yt.shape[1]}")
    print(f"{os.path.basename(p)}: {secs:.1f}s audio in {tot:.2f}s -> {secs / tot:.1f}x realtime on 1 thread")
