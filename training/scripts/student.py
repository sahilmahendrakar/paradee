"""Single-voice text-side student: width-scaled instances of Kokoro's own modules
(CustomAlbert, ProsodyPredictor, TextEncoder) with a learned constant style vector in place
of the voice input, plus a projection to the teacher's 512-channel asr space so the frozen
teacher decoder can consume its output."""
import torch, torch.nn as nn
from transformers import AlbertConfig
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder

PRESETS = {  # name: (albert hidden, albert layers, albert heads, hidden_dim, style_dim, n_layer)
    # "<name>+mlp" variants use a 2-layer nonlinear asr projection (the teacher asr features are rank ~430 for 99% variance)
    "xs": (128, 4, 4, 128, 16, 2),
    "s":  (256, 6, 4, 192, 32, 3),
    "m":  (384, 6, 6, 256, 32, 3),
    "l":  (512, 8, 8, 384, 64, 3),
}

class TextStudent(nn.Module):
    def __init__(self, preset="s", n_token=178, teacher_dim=512, max_dur=50):
        super().__init__()
        mlp = preset.endswith("+mlp"); preset = preset.replace("+mlp", "")
        ah, al, an, hd, sd, nl = PRESETS[preset]
        self.bert = CustomAlbert(AlbertConfig(vocab_size=n_token, hidden_size=ah, num_attention_heads=an,
                                              intermediate_size=ah * 3, num_hidden_layers=al,
                                              max_position_embeddings=512, embedding_size=min(128, ah), dropout=0.1))
        self.bert_encoder = nn.Linear(ah, hd)
        self.predictor = ProsodyPredictor(style_dim=sd, d_hid=hd, nlayers=nl, max_dur=max_dur, dropout=0.1)
        self.text_encoder = TextEncoder(channels=hd, kernel_size=5, depth=nl, n_symbols=n_token)
        self.asr_proj = (nn.Sequential(nn.Linear(hd, teacher_dim), nn.GELU(), nn.Linear(teacher_dim, teacher_dim)) if mlp
                         else nn.Linear(hd, teacher_dim))
        self.style = nn.Parameter(torch.randn(1, sd) * 0.1)  # the one voice

    def forward(self, ids, lengths, aln=None):
        """ids [B,T] long, lengths [B]. aln: optional teacher alignment [B,T,F] (teacher forcing).
        Returns dur [B,T] (pre-round sigmoid sums), F0 [B,2F], N [B,2F], asr_tok [B,512,T]."""
        B, T = ids.shape
        mask = torch.arange(T, device=ids.device)[None].expand(B, -1) >= lengths[:, None]
        s = self.style.expand(B, -1)
        bert = self.bert(ids, attention_mask=(~mask).int())
        d_en = self.bert_encoder(bert).transpose(-1, -2)
        d = self.predictor.text_encoder(d_en, s, lengths, mask)
        x, _ = self.predictor.lstm(d)
        dur = torch.sigmoid(self.predictor.duration_proj(x)).sum(-1)
        t_en = self.text_encoder(ids, lengths, mask)
        asr_tok = self.asr_proj(t_en.transpose(-1, -2)).transpose(-1, -2)
        if aln is None:
            return dur, None, None, asr_tok
        en = d.transpose(-1, -2) @ aln
        F0, N = self.predictor.F0Ntrain(en, s)
        return dur, F0, N, asr_tok

def alignment(pred_dur, F=None):
    """pred_dur [T] long -> [T, F] one-hot expansion matrix."""
    idx = torch.repeat_interleave(torch.arange(len(pred_dur), device=pred_dur.device), pred_dur)
    F = F or len(idx)
    a = torch.zeros(len(pred_dur), F, device=pred_dur.device); a[idx[:F], torch.arange(min(F, len(idx)), device=pred_dur.device)] = 1
    return a

if __name__ == "__main__":
    for p in list(PRESETS) + ["s+mlp"]:
        m = TextStudent(p); n = sum(x.numel() for x in m.parameters())
        parts = {k: sum(x.numel() for x in getattr(m, k).parameters()) for k in ["bert", "predictor", "text_encoder", "asr_proj"]}
        print(f"{p}: {n/1e6:.2f}M  " + "  ".join(f"{k} {v/1e6:.2f}M" for k, v in parts.items()))
