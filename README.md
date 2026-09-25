# Paradee

Paradee is a small English text-to-speech model. It has 8.07M parameters and is distilled from
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), and it speaks one voice, Kokoro's `af_heart`.

- **Small.** The int8 model is one 9 MB ONNX file, against 325 MB for Kokoro.
- **Fast.** It runs about 18x faster than real time on one CPU thread, with no GPU.
- **Close to its teacher.** It scores 4.41 on UTMOS (Kokoro: 4.52) and has the same word error rate (5.7%).

Paradee also runs inside the browser. It is the light voice in the
[Chickadee](https://github.com/sahilmahendrakar/chickadee) Chrome extension, where it needs no WebGPU and no download.

**Model files and audio samples:** [huggingface.co/sahilmahendrakar/Paradee-8M-v1.0](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0)

## Quick start

```bash
pip install git+https://github.com/sahilmahendrakar/paradee
python -m paradee "Paradee is a small voice that runs anywhere." -o hello.wav
```

From Python:

```python
from paradee import Paradee, SAMPLE_RATE
import soundfile as sf

tts = Paradee()   # downloads the 9 MB model from the Hugging Face Hub the first time
audio = tts("Paradee is a small voice that runs anywhere.")
sf.write("hello.wav", audio, SAMPLE_RATE)
```

Text is turned into phonemes with [misaki](https://github.com/hexgrad/misaki), Kokoro's own
grapheme-to-phoneme library, which is what the training data used. The first run also downloads
misaki's English dictionary data.

## How it compares

All numbers are on the same 200 held-out sentences, from the paper. Speed is on one CPU thread of
an Apple M4 Pro.

| Model (voice) | Params | File size | Speed | UTMOS | WER |
|---|---:|---:|---:|---:|---:|
| Kokoro-82M, teacher (af_heart) | 81.8M | 325 MB | 7.6x | 4.52 | 5.7% |
| **Paradee (af_heart)** | **8.07M** | **8.45 MB** | **25.0x** | **4.41** | **5.7%** |
| Kokoro-7M-Distill (af_msa) | 7.48M | 30.1 MB | 35.5x | 4.18 | 7.4% |
| Piper, en_US-lessac-medium | 15.7M | 63.2 MB | 15.4x | 4.36 | 8.8% |
| KittenTTS nano 0.8 (Bella) | 14.0M | 56.8 MB | 10.7x | 4.01 | 5.5% |

UTMOS is a neural network trained on human ratings. It predicts how natural a clip sounds, on a
scale from 1 to 5. WER is the share of words that Whisper (base) transcribes wrongly.

The paper's speed is for PyTorch. The released ONNX file (int8, with the phase filter included)
runs at about 18x real time in onnxruntime on one thread. Checked on the same 200 sentences, it
scores UTMOS 4.41 and WER 6.0% (this WER script normalizes text slightly differently from the paper's).

## How it works

Kokoro has two halves. A text side reads phonemes and predicts how long each one lasts, the pitch
and loudness over time, and a feature vector for each phoneme. A decoder turns those into audio.
Paradee is Kokoro's own code at smaller widths, trained half by half against the frozen teacher.

1. **Corpus.** Kokoro reads 12,000 WikiText-103 sentences (23.9 hours of audio), and every
   intermediate value is saved.
2. **Text side (4.23M parameters).** It learns to predict the teacher's durations, pitch, loudness
   and phoneme features directly.
3. **Decoder (3.85M parameters).** It learns to turn the teacher's saved values into the teacher's
   audio, first with spectrogram losses and then with adversarial training.
4. **Assembly.** The two halves are joined with no further training, and the weights are stored
   in int8.

A filter with no parameters then corrects the phase of voiced sound between 2 and 8 kHz (phase is
the timing of each frequency's wave). This removes a slight buzz that the small decoder otherwise
leaves. The ONNX file includes this filter.

The code for every step is in [`training/`](training/). It runs on one laptop.

## Using the ONNX file directly

The model is one ONNX graph that goes from token ids to audio:

- **Inputs:** `input_ids`, int64 `[1, T]`. These are phoneme ids from Kokoro's vocabulary (in
  `config.json` on the Hub), with a `0` pad token at each end, at most 512 in total. The second
  input is `speed`, float32 `[1]`, where 1.0 is normal speed.
- **Output:** `waveform`, float32 `[1, samples]` at 24 kHz.

[`web/misaki.js`](web/misaki.js) is for the browser. Browser phonemizers such as
[kokoro-js](https://github.com/hexgrad/kokoro/tree/main/kokoro.js) use eSpeak NG, which writes
some sounds differently from misaki. Paradee only learned misaki's spelling. Without this
conversion Whisper mishears about 31% of words, and with it about 2%.

## License

Apache 2.0, the same as Kokoro. The sentences in `training/data/` come from WikiText-103 and are
under CC BY-SA 3.0.

## Acknowledgements

Paradee is distilled from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad,
which builds on [StyleTTS 2](https://github.com/yl4579/StyleTTS2) and
[iSTFTNet](https://arxiv.org/abs/2203.02395). Phonemes come from
[misaki](https://github.com/hexgrad/misaki).

## Citation

```bibtex
@misc{mahendrakar2026paradee,
  title  = {Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model},
  author = {Mahendrakar, Sahil},
  year   = {2026},
  url    = {https://github.com/sahilmahendrakar/paradee}
}
```
