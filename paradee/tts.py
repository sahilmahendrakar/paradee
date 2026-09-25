"""Text -> phonemes (misaki, Kokoro's own G2P) -> token ids -> one ONNX graph -> 24 kHz audio."""
import json, re
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "sahilmahendrakar/Paradee-8M-v1.0"
REVISION = "v1.0"
SAMPLE_RATE = 24000
MAX_PHONEMES = 510   # the text side has 512 positions, two of which are the pad tokens at each end


class Paradee:
    """Load once, then call with text to get a float32 waveform at 24 kHz.

    quantized: use the 9 MB int8 file (default) instead of the 37 MB fp32 one. They sound the same.
    model_path / config_path: local files instead of downloading from the Hugging Face Hub.
    threads: CPU threads for onnxruntime. One thread already runs about 20x faster than real time.
    """

    def __init__(self, quantized=True, model_path=None, config_path=None, threads=1):
        name = "onnx/paradee_int8.onnx" if quantized else "onnx/paradee.onnx"
        model_path = model_path or hf_hub_download(REPO, name, revision=REVISION)
        config_path = config_path or hf_hub_download(REPO, "config.json", revision=REVISION)
        self.vocab = json.load(open(config_path, encoding="utf-8"))["vocab"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        from misaki import en, espeak   # the same G2P setup as kokoro.KPipeline(lang_code="a"), which made the training data
        self.g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")

    def phonemize(self, text):
        phonemes, _ = self.g2p(text)
        return phonemes

    def generate_from_phonemes(self, phonemes, speed=1.0):
        ids = [0] + [self.vocab[c] for c in phonemes if c in self.vocab][:MAX_PHONEMES] + [0]
        feed = {"input_ids": np.array([ids], dtype=np.int64), "speed": np.array([speed], dtype=np.float32)}
        return self.session.run(None, feed)[0][0]

    def __call__(self, text, speed=1.0):
        """speed > 1 speaks faster. Long text is read one sentence at a time."""
        parts = [self.generate_from_phonemes(ps, speed) for ps in self._chunks(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _chunks(self, text):
        for sentence in re.split(r"(?<=[.!?…])\s+|\n+", text.strip()):
            if not sentence.strip():
                continue
            ps = self.phonemize(sentence)
            while len(ps) > MAX_PHONEMES:   # a very long sentence: cut at the last space that fits
                cut = ps.rfind(" ", 0, MAX_PHONEMES)
                cut = cut if cut > 0 else MAX_PHONEMES
                yield ps[:cut]
                ps = ps[cut:].lstrip()
            if ps:
                yield ps
