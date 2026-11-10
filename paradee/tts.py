"""Text -> phonemes (misaki for English, Paradee's own G2P for Hindi, French and Spanish and for the English words
misaki has no entry for) -> token ids -> one ONNX graph -> 24 kHz audio."""
import json, re, unicodedata
from dataclasses import dataclass
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPOS = {"1.3": "sahilmahendrakar/Paradee-8M-v1.3", "1.2": "sahilmahendrakar/Paradee-8M-v1.2", "1.1": "sahilmahendrakar/Paradee-8M-v1.1", "1.0": "sahilmahendrakar/Paradee-8M-v1.0"}
# the voices of each version, in the order the model numbers them. 1.3 keeps 1.2's four voices at the same numbers
# and adds two Hindi voices, a French one and a Spanish one. 1.0 and 1.1 have af_heart only.
VOICES = {
    "1.3": ["af_heart", "bf_isabella", "am_puck", "bm_fable", "hf_alpha", "hm_omega", "ff_siwis", "em_alex"],
    "1.2": ["af_heart", "bf_isabella", "am_puck", "bm_fable"], "1.1": ["af_heart"], "1.0": ["af_heart"],
}
# A voice's first letter is its language, as in Kokoro: a = American English, b = British English, h = Hindi,
# f = French, e = Spanish. The values are the eSpeak languages Kokoro uses for the languages misaki doesn't cover.
LANGUAGES = {"a": "en-us", "b": "en-gb", "h": "hi", "f": "fr-fr", "e": "es"}
G2PS = ("paradee", "espeak")
SAMPLE_RATE = 24000
MAX_PHONEMES = 510   # the text side has 512 positions, two of which are the pad tokens at each end
# Inside every released graph (1.0, 1.1, 1.2): each token's length in frames of 600 samples, rounded. The frames
# add up to the waveform's length exactly. It is not a graph output; the loader adds it as one (_with_output).
DURATIONS = "/Clip_output_0"


def _with_output(model, name):
    """The ONNX file `model` (bytes) with the float tensor `name` as one more graph output, so the durations can
    be read without re-exporting the model and without the onnx package. An ONNX file is a protobuf: the graph
    is field 7 of the model and its outputs are field 12 of the graph, so one ValueInfoProto is appended to the
    graph and the graph's length prefix rewritten. Every other byte stays as it was."""
    def varint(n):
        out = bytearray()
        while True:
            out.append((n & 0x7F) | (0x80 if n > 0x7F else 0))
            n >>= 7
            if not n:
                return bytes(out)

    def read_varint(i):
        n = shift = 0
        while True:
            b = model[i]
            i += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                return n, i

    def field(num, payload):   # a length-delimited field
        return varint(num << 3 | 2) + varint(len(payload)) + payload

    i = 0
    while i < len(model):
        start = i
        key, i = read_varint(i)
        kind = key & 7
        if kind == 0:
            _, i = read_varint(i)
        elif kind == 1:
            i += 8
        elif kind == 5:
            i += 4
        elif kind == 2:
            n, body = read_varint(i)
            i = body + n
            if key >> 3 == 7:
                # ValueInfoProto{name, type: TypeProto{tensor_type: {elem_type: FLOAT}}}
                info = field(1, name.encode()) + field(2, field(1, b"\x08\x01"))
                return model[:start] + field(7, model[body:i] + field(12, info)) + model[i:]
        else:
            break
    raise ValueError("no graph found in the ONNX model")


def _phonemes(tokens):
    """misaki's tokens as the phoneme string the model reads, one space between words."""
    return "".join(t.phonemes + (" " if t.whitespace else "") for t in tokens).strip()


@dataclass
class _Word:
    """The parts of a misaki token that _phonemes, _token_chunks and _word_times read."""
    text: str
    phonemes: str
    whitespace: bool = True


def _espeak_tokens(text, phonemes):
    """eSpeak gives one phoneme string for a whole sentence, with a space between words. Split it back into
    words so the timestamps can use it like misaki's tokens. The words are paired with the words of the text in
    order. When the counts differ (a few percent of sentences: a number read as several words, an abbreviation
    spelled out) the words get no text, and with_timestamps spreads the text's words over the sentence instead."""
    parts = phonemes.split(" ")
    spoken = [i for i, p in enumerate(parts) if any(c.isalpha() for c in p)]
    words = _text_words(text)
    tokens = [_Word("", p) for p in parts]
    if len(spoken) == len(words):
        for i, w in zip(spoken, words):
            tokens[i].text = w
    if tokens:
        tokens[-1].whitespace = False
    return tokens, len(spoken) == len(words)


def _text_words(text):
    """The words of `text`, without the punctuation around them ("¿Dónde" -> "Dónde", "heures," -> "heures")."""
    def bare(w):
        while w and unicodedata.category(w[0])[0] == "P":
            w = w[1:]
        while w and unicodedata.category(w[-1])[0] == "P":
            w = w[:-1]
        return w
    return [b for b in map(bare, text.split()) if any(c.isalnum() for c in b)]


def _spread(words, start, end):
    """The text's words over [start, end], each getting time in proportion to its length in characters."""
    total = sum(len(w) for w in words)
    out, t = [], start
    for w in words:
        d = (end - start) * len(w) / total
        out.append((w, t, t + d))
        t += d
    return out


def _token_chunks(tokens):
    """misaki's tokens in runs whose phonemes fit the model, cut between words, as _chunks cuts text."""
    chunk = []
    for t in tokens:
        t.phonemes = t.phonemes or ""
        if chunk and len(_phonemes(chunk + [t])) > MAX_PHONEMES:
            yield chunk
            chunk = []
        chunk.append(t)
    if chunk:
        yield chunk


def _word_times(tokens, durations):
    """(text, start, end) in seconds for each word among `tokens`, from the frames of every token id, the pad at
    each end included. This is Kokoro's KPipeline.join_timestamps (kokoro/pipeline.py, Apache-2.0): it counts in
    half-frames, so the frames of a space are split between the words either side of it."""
    words = []
    if len(durations) < 3:
        return words
    left = right = 2 * max(0, int(durations[0]) - 3)
    i = 1
    for t in tokens:
        if i >= len(durations) - 1:
            break
        if not t.phonemes:
            if t.whitespace:
                i += 1
                left = right + int(durations[i])
                right = left + int(durations[i])
                i += 1
            continue
        j = i + len(t.phonemes)
        if j >= len(durations):
            break
        start = left / 80
        space = int(durations[j]) if t.whitespace else 0
        left = right + 2 * int(durations[i:j].sum()) + space
        right = left + space
        if any(c.isalnum() for c in t.text):   # punctuation has a duration too, but is not a word
            words.append((t.text, start, left / 80))
        i = j + (1 if t.whitespace else 0)
    return words


class Paradee:
    """Load once, then call with text to get a float32 waveform at 24 kHz.

    version: "1.3" (default, eight voices in English, Hindi, French and Spanish), "1.2" (the four English
    voices), "1.1" or "1.0" (af_heart only; 1.0 has the slower decoder).
    quantized: use the 8.5 MB int8 file (default) instead of the 35 MB fp32 one. They sound the same.
    model_path / config_path: local files instead of downloading from the Hugging Face Hub.
    threads: CPU threads for onnxruntime. One thread already runs about 60x faster than real time.
    g2p: "paradee" (default) turns text into phonemes with Paradee's own small G2P model (paradee/g2p.py), which
    gives eSpeak NG's phonemes without eSpeak. "espeak" uses eSpeak NG itself, as Kokoro does; it is GPL-3.0 and
    needs `pip install paradee[espeak]`.
    """

    def __init__(self, version="1.3", quantized=True, model_path=None, config_path=None, threads=1, g2p="paradee"):
        if version not in REPOS:
            raise ValueError(f"unknown version {version!r}; choose one of {', '.join(REPOS)}")
        if g2p not in G2PS:
            raise ValueError(f"unknown g2p {g2p!r}; choose one of {', '.join(G2PS)}")
        self.g2p_name = g2p
        repo, revision = REPOS[version], f"v{version}"
        self.version, self.voices = version, VOICES[version]
        name = "onnx/paradee_int8.onnx" if quantized else "onnx/paradee.onnx"
        model_path = model_path or hf_hub_download(repo, name, revision=revision)
        config_path = config_path or hf_hub_download(repo, "config.json", revision=revision)
        self.vocab = json.load(open(config_path, encoding="utf-8"))["vocab"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        with open(model_path, "rb") as f:
            model = _with_output(f.read(), DURATIONS)
        self.session = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        self._g2p, self._own = {}, None

    def g2p(self, voice="af_heart"):
        """The G2P for the voice's language (its first letter): misaki's American or British English for a and b,
        and phonemes in eSpeak's style for h (Hindi), f (French) and e (Spanish). With g2p="espeak" this is the
        same setup as kokoro.KPipeline, which made the training data; with g2p="paradee" Paradee's own model
        stands in for eSpeak, both for the three languages and for the English words misaki has no entry for.
        Each is loaded the first time it is needed. Like misaki's, it returns (phonemes, tokens), and tokens is
        None for h, f and e."""
        lang = voice[0]
        if lang not in LANGUAGES:
            raise ValueError(f"no language for voice {voice!r}")
        if lang not in self._g2p:
            if self.g2p_name == "paradee":
                from .g2p import G2P
                self._own = self._own or G2P()
                own = self._own
                if lang in "ab":
                    from misaki import en
                    british = lang == "b"
                    self._g2p[lang] = en.G2P(trf=False, british=british, fallback=own.english_fallback(british), unk="")
                else:
                    code = {"h": "hi", "f": "fr", "e": "es"}[lang]
                    self._g2p[lang] = lambda text: (own(text, code), None)
            else:
                try:
                    from misaki import en, espeak
                except ImportError as e:
                    raise ImportError('g2p="espeak" needs eSpeak NG: pip install "paradee[espeak]"') from e
                if lang in "ab":
                    british = lang == "b"
                    self._g2p[lang] = en.G2P(trf=False, british=british, fallback=espeak.EspeakFallback(british=british), unk="")
                else:
                    self._g2p[lang] = espeak.EspeakG2P(language=LANGUAGES[lang])
        return self._g2p[lang]

    def phonemize(self, text, voice="af_heart"):
        """Phonemes in the voice's language: British for bf_isabella and bm_fable, American for af_heart and
        am_puck, Hindi for hf_alpha and hm_omega, French for ff_siwis, Spanish for em_alex."""
        phonemes, _ = self.g2p(voice)(text)
        return phonemes

    def check_voice(self, voice):
        if voice not in self.voices:
            raise ValueError(f"Paradee {self.version} has no voice {voice!r}; its voices are {', '.join(self.voices)}")

    def generate_from_phonemes(self, phonemes, speed=1.0, voice="af_heart"):
        return self._run(phonemes, speed, voice)[0]

    def _run(self, phonemes, speed, voice):
        """(waveform, frames of every token id), for one run of the model."""
        self.check_voice(voice)
        ids = [0] + [self.vocab[c] for c in phonemes if c in self.vocab][:MAX_PHONEMES] + [0]
        feed = {"input_ids": np.array([ids], dtype=np.int64), "speed": np.array([speed], dtype=np.float32)}
        if len(self.voices) > 1:
            feed["voice"] = np.array([self.voices.index(voice)], dtype=np.int64)
        waveform, durations = self.session.run(None, feed)
        return waveform[0], np.rint(durations.reshape(-1)).astype(np.int64)

    def __call__(self, text, speed=1.0, voice="af_heart"):
        """speed > 1 speaks faster. voice is one of self.voices. Long text is read one sentence at a time."""
        self.check_voice(voice)
        parts = [self.generate_from_phonemes(ps, speed, voice) for ps in self._chunks(text, voice)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def with_timestamps(self, text, speed=1.0, voice="af_heart"):
        """Like calling the model, but also says when each word is spoken, for captions or for highlighting the
        word being read. Returns (audio, words): each word is (text, start, end) in seconds of that audio, and
        punctuation is left out. The times come from the durations the model made the audio from."""
        self.check_voice(voice)
        g2p = self.g2p(voice)
        parts, words, offset = [], [], 0.0
        for sentence in self._sentences(text):
            phonemes, tokens = g2p(sentence)
            paired = True
            if tokens is None:   # eSpeak (Hindi, French, Spanish) gives no tokens
                tokens, paired = _espeak_tokens(sentence, phonemes)
            for t in tokens:   # the model never sees symbols outside its vocabulary (Spanish ¿ ¡), so don't count them
                t.phonemes = "".join(c for c in (t.phonemes or "") if c in self.vocab)
            start = offset
            for chunk in _token_chunks(tokens):
                audio, durations = self._run(_phonemes(chunk), speed, voice)
                if paired:
                    words += [(w, offset + s, offset + e) for w, s, e in _word_times(chunk, durations)]
                offset += len(audio) / SAMPLE_RATE
                parts.append(audio)
            if not paired:
                words += _spread(_text_words(sentence), start, offset)
        return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), words

    @staticmethod
    def _sentences(text):
        # "।" (and "॥") end a sentence in Hindi, often with no space after them
        return (s for s in re.split(r"(?<=[.!?…])\s+|(?<=[।॥])\s*|\n+", text.strip()) if s.strip())

    def _chunks(self, text, voice="af_heart"):
        for sentence in self._sentences(text):
            ps = self.phonemize(sentence, voice)
            while len(ps) > MAX_PHONEMES:   # a very long sentence: cut at the last space that fits
                cut = ps.rfind(" ", 0, MAX_PHONEMES)
                cut = cut if cut > 0 else MAX_PHONEMES
                yield ps[:cut]
                ps = ps[cut:].lstrip()
            if ps:
                yield ps
