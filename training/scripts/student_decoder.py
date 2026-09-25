"""Acoustic-side student: Kokoro's Decoder with the hard-coded 1024/512 widths made configurable,
generator sized via the same istftnet config keys."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import weight_norm
from kokoro.istftnet import Decoder, AdainResBlk1d, Generator, SourceModuleHnNSF, SineGen

DEC_PRESETS = {  # name: (adain width, upsample_initial_channel, resblock kernels)
    "teacher": (1024, 512, [3, 7, 11]),
    "A": (256, 128, [3, 7, 11]),
    "B": (256, 128, [3, 7]),
    "C": (192, 96, [3, 7]),
    "D": (128, 64, [3, 7]),
    "E": (128, 64, [3, 5]),
    "F": (96, 48, [3, 5]),
    "W": (192, 256, [3, 7, 11]),   # wider vocoder final stage (64 ch), thinner AdaIN blocks
    "W2": (128, 256, [3, 7]),
    "L": (384, 192, [3, 7, 11]),   # 7.86M: capacity probe, 1.5x wider everywhere than A
}

class RichSource(SourceModuleHnNSF):
    """Teacher's source module with more harmonics; harmonics above 11 kHz are muted instead of aliasing."""
    def forward(self, x):
        with torch.no_grad():
            sine, uv, _ = self.l_sin_gen(x); k = torch.arange(1, sine.shape[-1] + 1, device=x.device)
            sine = sine * (x * k < 11000)
        return self.l_tanh(self.l_linear(sine)), torch.randn_like(uv) * self.sine_amp / 3, uv

class PhaseSineGen(SineGen):
    """SineGen that also keeps each harmonic's instantaneous phase (same RNG use, same output)."""
    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device); rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        self.phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
        return torch.sin(self.phase)

class BandGenerator(Generator):
    """Kokoro generator plus two optional student-only inputs/outputs, both zero-initialised so a
    checkpoint loads to the same function:
    bands: every harmonic's phase as its own (cos, sin) pair at the 4800 fps head rate, fed to each
    stage next to the mixed 1-channel source STFT, so upper harmonics are no longer summed together.
    full_phase: an unbounded additive phase term, since the head's sin(x) alone confines phase to +-1 rad."""
    def __init__(self, *a, harmonics=9, bands=False, full_phase=False, **kw):
        super().__init__(*a, **kw); self.bands, self.full_phase, self.hop = bands, full_phase, a[-1]
        if bands:
            self.band_convs = nn.ModuleList([nn.Conv1d(2 * harmonics, c.out_channels, c.kernel_size, c.stride, c.padding) for c in self.noise_convs])
            for c in self.band_convs: nn.init.zeros_(c.weight); nn.init.zeros_(c.bias)
        if full_phase:
            ch = self.conv_post.in_channels; self.conv_phase = nn.Conv1d(ch, self.post_n_fft // 2 + 1, 7, 1, padding=3)
            nn.init.zeros_(self.conv_phase.weight); nn.init.zeros_(self.conv_phase.bias)

    def forward(self, x, s, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, _, uv = self.m_source(f0)
            har_spec, har_phase = self.stft.transform(har_source.transpose(1, 2).squeeze(1))
            har = torch.cat([har_spec, har_phase], dim=1)
            if self.bands:
                ph = self.m_source.l_sin_gen.phase; k = torch.arange(1, ph.shape[-1] + 1, device=ph.device)
                g = uv * (f0 * k < 11000)
                q = torch.cat([torch.cos(ph) * g, torch.sin(ph) * g], -1).transpose(1, 2)
                q = F.pad(q, (0, 1), mode="replicate")[..., ::self.hop]   # STFT frame centres
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            if self.bands: x_source = x_source + self.band_convs[i](q)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1: x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                xs = self.resblocks[i * self.num_kernels + j](x, s) if xs is None else xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        h = F.leaky_relu(x); x = self.conv_post(h); nb = self.post_n_fft // 2 + 1
        spec = torch.exp(x[:, :nb, :]); phase = torch.sin(x[:, nb:, :])
        if self.full_phase: phase = phase + self.conv_phase(h)
        return self.stft.inverse(spec, phase)

class StudentDecoder(Decoder):
    def __init__(self, preset="C", dim_in=512, style_dim=16, dim_out=80, disable_complex=False, harmonics=9, bands=False, full_phase=False):
        W, uic, kernels = DEC_PRESETS[preset]
        nn.Module.__init__(self)
        self.encode = AdainResBlk1d(dim_in + 2, W, style_dim)
        self.decode = nn.ModuleList([AdainResBlk1d(W + 2 + 64, W, style_dim) for _ in range(3)]
                                    + [AdainResBlk1d(W + 2 + 64, uic, style_dim, upsample=True)])
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(dim_in, 64, kernel_size=1)))
        gargs = (style_dim, kernels, [10, 6], uic, [[1, 3, 5]] * len(kernels), [20, 12], 20, 5)
        self.generator = (BandGenerator(*gargs, harmonics=harmonics, bands=bands, full_phase=full_phase, disable_complex=disable_complex)
                          if bands or full_phase else Generator(*gargs, disable_complex=disable_complex))
        self.style = nn.Parameter(torch.randn(1, style_dim) * 0.1)
        if harmonics != 9: self.generator.m_source = RichSource(24000, 300, harmonic_num=harmonics - 1, voiced_threshod=10)
        if bands: self.generator.m_source.l_sin_gen.__class__ = PhaseSineGen

    def forward(self, asr, F0_curve, N, s=None):
        return super().forward(asr, F0_curve, N, self.style.expand(asr.shape[0], -1))

if __name__ == "__main__":
    from torch.utils.flop_counter import FlopCounterMode
    T = 174; asr = torch.randn(1, 512, T); F0 = torch.rand(1, 2 * T) * 200 + 80; N = torch.randn(1, 2 * T)
    print(f"{'preset':8s} {'params':>8s} {'gen':>7s} {'adain':>7s} {'GFLOP/s':>8s} {'gen%':>5s}")
    for p in DEC_PRESETS:
        m = StudentDecoder(p, style_dim=128 if p == "teacher" else 16).eval().requires_grad_(False)
        fc = FlopCounterMode(display=False, depth=2)
        with fc: out = m(asr, F0, N)
        c = {k: sum(v.values()) for k, v in fc.get_flop_counts().items()}; g = c["Global"]; gen = c.get("StudentDecoder.generator", 0)
        secs = out.shape[-1] / 24000
        np_ = lambda mod: sum(x.numel() for x in mod.parameters()) / 1e6
        print(f"{p:8s} {np_(m):7.2f}M {np_(m.generator):6.2f}M {np_(m)-np_(m.generator):6.2f}M {g/secs/1e9:8.2f} {gen/g*100:4.0f}%")

def load_student_decoder(preset, path, device="cpu", **kw):
    """Load a StudentDecoder checkpoint. Matched-excitation checkpoints carry no source-mixing weights
    (the module was replaced during training); those are taken from the teacher, as in training."""
    import torch
    sd = torch.load(path, map_location=device)["model"]
    lw = sd.get("generator.m_source.l_linear.weight")
    if lw is not None: kw.setdefault("harmonics", lw.shape[1])
    if "generator.band_convs.0.weight" in sd: kw.setdefault("bands", True)
    if "generator.conv_phase.weight" in sd: kw.setdefault("full_phase", True)
    m = StudentDecoder(preset, **kw).to(device).eval()
    res = m.load_state_dict(sd, strict=False)
    if any("m_source" in k for k in res.missing_keys):
        from common import load_teacher
        _p, t, _ = load_teacher(device=device, fuse=False)
        m.generator.m_source.load_state_dict(t.decoder.generator.m_source.state_dict()); del t
    return m
