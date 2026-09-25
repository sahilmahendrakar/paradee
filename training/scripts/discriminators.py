"""HiFi-GAN style multi-period discriminator + multi-resolution spectrogram discriminator,
LS-GAN losses and feature matching. Compact versions sized for MPS training."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import weight_norm

class PeriodD(nn.Module):
    def __init__(self, period, ch=(32, 64, 128, 256), mult=1):
        ch = tuple(c * mult for c in ch)
        super().__init__(); self.period = period; c = 1; self.convs = nn.ModuleList()
        for o in ch: self.convs.append(weight_norm(nn.Conv2d(c, o, (5, 1), (3, 1), padding=(2, 0)))); c = o
        self.post = weight_norm(nn.Conv2d(c, 1, (3, 1), 1, padding=(1, 0)))
    def forward(self, x):
        b, t = x.shape; p = self.period
        if t % p: x = F.pad(x, (0, p - t % p), "reflect")
        x = x.view(b, 1, -1, p); feats = []
        for c in self.convs: x = F.leaky_relu(c(x), 0.1); feats.append(x)
        x = self.post(x); feats.append(x); return x.flatten(1), feats

class ResD(nn.Module):
    def __init__(self, n_fft, ch=(32, 64, 128, 128), mult=1):
        ch = tuple(c * mult for c in ch)
        super().__init__(); self.n_fft = n_fft; self.hop = n_fft // 4; c = 1; self.convs = nn.ModuleList()
        for o in ch: self.convs.append(weight_norm(nn.Conv2d(c, o, (3, 9), (1, 2), padding=(1, 4)))); c = o
        self.post = weight_norm(nn.Conv2d(c, 1, (3, 3), 1, padding=(1, 1)))
    def forward(self, x):
        w = torch.hann_window(self.n_fft, device=x.device)
        s = torch.stft(x, self.n_fft, self.hop, self.n_fft, window=w, return_complex=True).abs()[:, None]  # [B,1,F,T]
        feats = []
        for c in self.convs: s = F.leaky_relu(c(s), 0.1); feats.append(s)
        s = self.post(s); feats.append(s); return s.flatten(1), feats

class Discriminators(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), n_ffts=(512, 1024, 2048), mult=1):
        super().__init__()
        self.ds = nn.ModuleList([PeriodD(p, mult=mult) for p in periods] + [ResD(n, mult=mult) for n in n_ffts])
    def forward(self, x): return [d(x) for d in self.ds]

def d_loss(D, real, fake):
    loss = 0
    for (r, _), (f, _) in zip(D(real), D(fake.detach())):
        loss = loss + ((r - 1) ** 2).mean() + (f ** 2).mean()
    return loss

def g_loss(D, real, fake):
    adv, fm = 0, 0
    for (r, rf), (f, ff) in zip(D(real), D(fake)):
        adv = adv + ((f - 1) ** 2).mean()
        fm = fm + sum(F.l1_loss(a, b.detach()) for a, b in zip(ff, rf)) / len(ff)
    return adv, fm
