"""Shared teacher loading, weight-norm fusion, fake quantization and metrics."""
import warnings
import numpy as np, torch, torchaudio
warnings.filterwarnings("ignore")
from kokoro import KPipeline
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

SENTS = [
 "The chickadee is a small songbird found across North America.",
 "In 1998, the company reported revenue of 4.2 billion dollars, up 17 percent from the year before.",
 "Would you rather read the article now, or should I save it for later?",
 "Despite its size, it survives harsh winters by caching thousands of seeds and remembering where each one is hidden.",
 "Dr. Smith arrived at 3 p.m. on Tuesday, carrying an umbrella, a laptop, and a very large sandwich.",
]

def load_teacher(device="cpu", voice="af_heart", fuse=True):
    pipe = KPipeline(lang_code="a", device=device, repo_id="hexgrad/Kokoro-82M")
    model = pipe.model.eval()
    pack = pipe.load_voice(voice)
    items = []
    for s in SENTS:
        ps, _ = pipe.g2p(s); items.append((ps, pack[len(ps) - 1]))
    if fuse:
        for _, m in model.named_modules():
            try: remove_weight_norm(m)
            except Exception: pass
            if is_parametrized(m, "weight"):
                try: remove_parametrizations(m, "weight", leave_parametrized=True)
                except Exception: pass
    return pipe, model, items

_mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80)
def logmel(x): return torch.log(_mel(x).clamp_min(1e-5))

def dtw_l1(a, b):
    """Mean L1 between log-mel frames [n_mels, T] along a DTW path (timing-insensitive)."""
    a, b = a.T.numpy(), b.T.numpy()
    C = np.abs(a[:, None, :] - b[None, :, :]).mean(-1)
    n, m = C.shape; D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, 0
    while i > 0 and j > 0:
        path += 1; k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return D[n, m] / path

def rmsnorm(x): return x / x.pow(2).mean().sqrt().clamp_min(1e-8)

def compare(refs, outs):
    """Returns (raw log-mel L1, DTW log-mel L1, mean |duration delta| ms). Gain-invariant (RMS-normalised)."""
    raw, dtw, ld = [], [], []
    for r, o in zip(refs, outs):
        if torch.isnan(o).any(): return float("nan"), float("nan"), float("nan")
        r, o = rmsnorm(r), rmsnorm(o)
        k = min(len(r), len(o)); lr, lo = logmel(r), logmel(o)
        raw.append((lr[:, :k // 256 + 1] - lo[:, :k // 256 + 1]).abs().mean().item())
        dtw.append(dtw_l1(lr, lo)); ld.append(abs(len(o) - len(r)) / 24)
    return float(np.mean(raw)), float(np.mean(dtw)), float(np.mean(ld))

def fq(w, bits, group=None):
    """Symmetric fake quantization, per output channel (dim 0) or per group along the rest."""
    if w.dim() < 2: return w
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    if group:
        pad = (-flat.shape[1]) % group
        g = torch.nn.functional.pad(flat, (0, pad)).reshape(flat.shape[0], -1, group)
        s = g.abs().amax(2, keepdim=True).clamp_min(1e-8) / qmax
        out = (torch.round(g / s).clamp(-qmax, qmax) * s).reshape(flat.shape[0], -1)[:, :flat.shape[1]]
    else:
        s = flat.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
        out = torch.round(flat / s).clamp(-qmax, qmax) * s
    return out.reshape(w.shape)

def cpu_source_stft(decoder):
    """Make an iSTFTNet decoder device-canonical: the harmonic-source STFT's raw phase feature sits on
    the ±pi branch cut for strong harmonic bins and flips sign between CPU and MPS, changing the output
    by ~0.4 DTW log-mel. Computing just that STFT on CPU makes MPS match CPU to 1e-5."""
    g = decoder.generator; st = g.stft; orig = type(st).transform
    def transform(x):
        mag, ph = orig(st, x.cpu()); return mag.to(x.device), ph.to(x.device)
    st.transform = transform
    return decoder
