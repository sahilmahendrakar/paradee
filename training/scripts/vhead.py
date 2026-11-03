"""Frame-rate source-filter head (the Vocos idea with a harmonic source) in place of Kokoro's upsampling
iSTFTNet generator. Decoder A's AdaIN trunk is kept (80 frames a second). The head never upsamples: a few
ConvNeXt blocks at 80 fps predict, for every STFT bin of 2 frames per trunk frame (n_fft 600, hop 150),
a gain and a phase offset applied to the STFT of a clean sine bank built from the F0 curve (harmonic k has
phase k*phi), plus a gain for white noise. Harmonics therefore arrive phase-locked to the pitch by
construction, and the network only learns the spectral envelope and a slow phase correction.
usage: vhead.py   prints params and FLOPs next to decoder A"""
import torch, torch.nn as nn, torch.nn.functional as F
from student_decoder import StudentDecoder
SR, FHOP = 24000, 300   # F0 frames are 300 samples

def sine_bank(f0, K=64, c0=None):
    """f0 [B, frames] (80 fps) -> [B, samples] sum of K harmonics with Kokoro's fundamental phase (frame-rate
    cumulative phase, linearly interpolated between frame centres), harmonics muted above 11 kHz and where
    unvoiced. The running phase is accumulated in float64 and wrapped per frame so harmonic 64 stays exact.
    c0 [B]: phase (cycles) accumulated before this crop, so a segment's bank matches the whole sentence's."""
    B, n = f0.shape; dev = f0.device
    r = (f0.cpu().double() * FHOP / SR)                                  # cycles per frame
    c = ((torch.cumsum(r, 1) + (0 if c0 is None else c0.cpu().double()[:, None])) % 1).float().to(dev); d = F.pad(r[:, 1:], (0, 1)).float().to(dev)   # wrapped start, advance to next centre
    p = (torch.arange(n * FHOP, device=dev) + 0.5) / FHOP - 0.5; i = p.floor().clamp(0, n - 1).long(); fr = (p - i).clamp(0, 1)
    cyc = c[:, i] + fr * d[:, i]                                         # [B, samples] fundamental phase in cycles
    f0s = f0.repeat_interleave(FHOP, 1); k = torch.arange(1, K + 1, device=dev)[None, :, None]
    amp = ((f0s[:, None] * k < 11000) & (f0s[:, None] > 10)).float()
    return (torch.sin(2 * torch.pi * k * cyc[:, None]) * amp).sum(1) * 0.1, cyc, (f0s > 10).float()

class ConvNeXt(nn.Module):
    def __init__(self, dim, mult=3):
        super().__init__(); self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim); self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mult); self.pw2 = nn.Linear(dim * mult, dim); self.gamma = nn.Parameter(torch.full((dim,), 1e-2))
    def forward(self, x):
        h = self.norm(self.dw(x).transpose(1, 2)); return x + (self.gamma * self.pw2(F.gelu(self.pw1(h)))).transpose(1, 2)

class SourceFilterHead(nn.Module):
    def __init__(self, in_ch=128, dim=160, blocks=4, n_fft=600, hop=150, K=64, mod=2, mb=33):
        """mod: the noise part can also pulse with the pitch: extra sources noise*cos(m*phi), noise*sin(m*phi) for m = 1..mod,
        each with a signed gain relative to the plain noise gain, predicted at mb points across frequency (zero-init)."""
        super().__init__(); self.n_fft, self.hop, self.K, self.up, self.nb, self.mod, self.mb = n_fft, hop, K, FHOP // hop, n_fft // 2 + 1, mod, mb
        if mod: self.out_mod = nn.Linear(dim, self.up * 2 * mod * mb); nn.init.zeros_(self.out_mod.weight); nn.init.zeros_(self.out_mod.bias)
        self.inp = nn.Conv1d(in_ch, dim, 7, padding=3); self.norm0 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ConvNeXt(dim) for _ in range(blocks)]); self.norm1 = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, self.up * self.nb * 3)   # per bin: harmonic log-gain, harmonic phase offset, noise log-gain
        nn.init.trunc_normal_(self.out.weight, std=0.02)
        with torch.no_grad(): b = self.out.bias.view(self.up, 3, self.nb); b[:, 0] = -3.0; b[:, 1] = 0.0; b[:, 2] = -6.0
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
    def stft(self, x): return torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True)
    def forward(self, x, f0, noise=None, c0=None):
        B, _, n = x.shape
        h = self.norm0(self.inp(x).transpose(1, 2)).transpose(1, 2)
        for b in self.blocks: h = b(h)
        hn = self.norm1(h.transpose(1, 2)); o = self.out(hn).view(B, n, self.up, 3, self.nb).permute(0, 3, 4, 1, 2).reshape(B, 3, self.nb, n * self.up)
        o = torch.cat([o, o[..., -1:]], -1)                # centred STFT has one more frame
        gh, ph, gn = torch.exp(o[:, 0].clamp(max=6)), o[:, 1], torch.exp(o[:, 2].clamp(max=6))
        with torch.no_grad():
            e, cyc, uv = sine_bank(f0, self.K, c0); E = self.stft(e)
            nz = torch.randn_like(e) if noise is None else noise; Nz = self.stft(nz)
            Nm = [self.stft(nz * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, self.mod + 1) for fn in (torch.cos, torch.sin)]
        self.last = dict(E=E, gh=gh, ph=ph, gn=gn)
        if self.mod:
            a = self.out_mod(hn).view(B, n, self.up, 2 * self.mod, self.mb).permute(0, 3, 1, 2, 4).reshape(B * 2 * self.mod, n * self.up, self.mb)
            a = F.interpolate(a, size=self.nb, mode="linear", align_corners=True).view(B, 2 * self.mod, n * self.up, self.nb).transpose(2, 3)
            a = torch.cat([a, a[..., -1:]], -1); self.last["a"] = a
            Nz = Nz + sum(a[:, j] * Nm[j] for j in range(2 * self.mod))
        Y = E * torch.polar(gh, ph) + Nz * gn
        return torch.istft(Y, self.n_fft, self.hop, window=self.window, length=n * FHOP)

def warp_matrix(nb, G, sr=SR):
    """[nb, G] linear-interpolation weights from G points spaced evenly on a mel-like axis to nb linear STFT bins."""
    f = torch.linspace(0, sr / 2, nb); mel = lambda x: torch.log1p(x / 700); pos = mel(f) / mel(torch.tensor(sr / 2.0)) * (G - 1)
    i = pos.floor().clamp(0, G - 2).long(); w = pos - i; M = torch.zeros(nb, G); M[torch.arange(nb), i] = 1 - w; M[torch.arange(nb), i + 1] += w; return M

class SourceFilterHead2(SourceFilterHead):
    """Second design: every part has one job. The harmonic part is a smooth filter on the sine bank: gain and phase are
    predicted at G points across frequency (mel-spaced) and interpolated, plus one pulse shift per frame (a delay of the
    whole pulse, as a fraction of the pitch period), so it cannot scramble individual bins into fake noise. Anything
    noise-like has to come from the noise part (plain + pulsing with the pitch), whose gains are also on the G points."""
    def __init__(self, in_ch=128, dim=160, blocks=4, n_fft=600, hop=150, K=64, mod=2, G=80, burst=True, Gb=24):
        """burst: a second noise source shaped through a short STFT (150-sample window, 75 hop: 3 ms steps, 4 per trunk
        frame) with gains at Gb points, so consonant bursts and onsets are not smeared over the 25 ms main window."""
        nn.Module.__init__(self); self.burst, self.Gb = burst, Gb
        if burst:
            self.out_b = nn.Linear(dim, 4 * Gb); nn.init.trunc_normal_(self.out_b.weight, std=0.02); nn.init.constant_(self.out_b.bias, -7.0)
            self.register_buffer("wb", torch.hann_window(150), persistent=False); self.register_buffer("Mb", warp_matrix(76, Gb), persistent=False)
        self.n_fft, self.hop, self.K, self.up, self.nb, self.mod, self.G = n_fft, hop, K, FHOP // hop, n_fft // 2 + 1, mod, G
        self.inp = nn.Conv1d(in_ch, dim, 7, padding=3); self.norm0 = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ConvNeXt(dim) for _ in range(blocks)]); self.norm1 = nn.LayerNorm(dim)
        self.C = 4 + 2 * mod                               # harmonic log-gain, phase cos, phase sin, noise log-gain, pulsing-noise gains
        self.out = nn.Linear(dim, self.up * (self.C * G + 1)); nn.init.trunc_normal_(self.out.weight, std=0.02)
        with torch.no_grad():
            b = self.out.bias.view(self.up, -1); b.zero_(); g = b[:, :self.C * G].view(self.up, self.C, G); g[:, 0] = -3.0; g[:, 1] = 1.0; g[:, 3] = -6.0
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False); self.register_buffer("M", warp_matrix(self.nb, G), persistent=False)
    def forward(self, x, f0, noise=None, c0=None):
        B, _, n = x.shape; T = n * self.up
        h = self.norm0(self.inp(x).transpose(1, 2)).transpose(1, 2)
        for b in self.blocks: h = b(h)
        hn = self.norm1(h.transpose(1, 2)); o = self.out(hn).view(B, T, self.C * self.G + 1); o = torch.cat([o, o[:, -1:]], 1)   # centred STFT has one more frame
        g = (o[..., :-1].reshape(B, T + 1, self.C, self.G) @ self.M.T).permute(0, 2, 3, 1)                                 # [B, C, nb, T+1]
        gh, gn = torch.exp(g[:, 0].clamp(max=6)), torch.exp(g[:, 3].clamp(max=6)); u = torch.complex(g[:, 1], g[:, 2]); u = u / u.abs().clamp_min(1e-4)
        with torch.no_grad():
            e, cyc, uv = sine_bank(f0, self.K, c0); E = self.stft(e)
            nz = torch.randn_like(e) if noise is None else noise; Nz = self.stft(nz)
            Nm = [self.stft(nz * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, self.mod + 1) for fn in (torch.cos, torch.sin)]
            f0f = torch.cat([f0.repeat_interleave(self.up, 1), f0[:, -1:]], 1).clamp_min(60)                               # per STFT frame
            ratio = torch.linspace(0, SR / 2, self.nb, device=x.device)[None, :, None] / f0f[:, None]                      # harmonic number of each bin
        shift = 0.5 * torch.tanh(o[..., -1])[:, None]                                                                      # pulse shift, cycles
        Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
        self.last = dict(E=E, gh=gh, ph=Hh.angle(), gn=gn, a=g[:, 4:], shift=shift, cyc=cyc, uv=uv)
        Nz = Nz + sum(g[:, 4 + j] * Nm[j] for j in range(2 * self.mod))
        y = torch.istft(E * Hh + Nz * gn, self.n_fft, self.hop, window=self.window, length=n * FHOP)
        if self.burst:
            gb = torch.exp((self.out_b(hn).view(B, n * 4, self.Gb) @ self.Mb.T).clamp(max=6)).transpose(1, 2); gb = torch.cat([gb, gb[..., -1:]], -1)
            with torch.no_grad(): Nb = torch.stft(nz.flip(-1), 150, 75, window=self.wb, return_complex=True)
            self.last["gb"] = gb; y = y + torch.istft(Nb * gb, 150, 75, window=self.wb, length=n * FHOP)
        return y

class VDecoder(nn.Module):
    """Decoder A's AdaIN trunk + the frame-rate head."""
    def __init__(self, preset="A", head="v1", n_voices=1, **kw):
        super().__init__(); t = StudentDecoder(preset); uic = t.generator.conv_post.in_channels * 4
        for k in ("encode", "decode", "F0_conv", "N_conv", "asr_res"): setattr(self, k, getattr(t, k))
        self.style = nn.Parameter(t.style.data.repeat(n_voices, 1)); self.head = (SourceFilterHead2 if head == "v2" else SourceFilterHead)(in_ch=uic, **kw)
    def forward(self, asr, F0_curve, N, noise=None, c0=None, voice=None):
        s = self.style[voice] if voice is not None else self.style[:1].expand(asr.shape[0], -1)
        F0 = self.F0_conv(F0_curve.unsqueeze(1)); Nn = self.N_conv(N.unsqueeze(1))
        x = self.encode(torch.cat([asr, F0, Nn], 1), s); asr_res = self.asr_res(asr); res = True
        for block in self.decode:
            if res: x = torch.cat([x, asr_res, F0, Nn], 1)
            x = block(x, s)
            if block.upsample_type != "none": res = False
        return self.head(x, F0_curve, noise, c0)
    def load_trunk(self, path):
        sd = {k: v for k, v in torch.load(path, map_location="cpu")["model"].items() if not k.startswith(("generator.", "head."))}
        res = self.load_state_dict(sd, strict=False); assert all(k.startswith("head.") for k in res.missing_keys) and not res.unexpected_keys, res
    def load_ckpt(self, path, device="cpu"):
        """Load a vhead checkpoint; one from before the pulsing-noise sources gets them zero-initialised (same function)."""
        st = torch.load(path, map_location=device)["model"]; w = st["style"]
        if w.shape[0] < self.style.shape[0]: st["style"] = torch.cat([w, w[:1].repeat(self.style.shape[0] - w.shape[0], 1)])   # new voices start from voice 0
        res = self.load_state_dict(st, strict=False); assert all("out_mod" in k or "out_b" in k for k in res.missing_keys) and not res.unexpected_keys, res
        if any("out_b" in k for k in res.missing_keys): self.head.burst = False   # checkpoint from before the short-window noise path

def HV(path):
    """Which head design a checkpoint holds (v2's output layer is the small one)."""
    return "v2" if torch.load(path, map_location="cpu")["model"]["head.out.weight"].shape[0] < 1500 else "v1"

if __name__ == "__main__":
    from torch.utils.flop_counter import FlopCounterMode
    T = 174; asr = torch.randn(1, 512, T); F0 = torch.rand(1, 2 * T) * 200 + 80; N = torch.randn(1, 2 * T)
    np_ = lambda m: sum(x.numel() for x in m.parameters()) / 1e6
    for name, m in [("decoder A", StudentDecoder("A")), ("A trunk + head", VDecoder()), ("A trunk + head2", VDecoder(head="v2"))]:
        m = m.eval().requires_grad_(False); fc = FlopCounterMode(display=False)
        with fc: out = m(asr, F0, N)
        hp = np_(m.head) if hasattr(m, "head") else np_(m.generator)
        print(f"{name:16s} {np_(m):5.2f}M params (head/generator {hp:.2f}M)  {fc.get_total_flops() / (out.shape[-1] / SR) / 1e9:6.2f} GFLOP/s  out {tuple(out.shape)}")
