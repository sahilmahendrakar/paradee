# Paradee

Paradee is a small English text-to-speech model. It has 7.88M parameters and is distilled from
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), and it speaks four of Kokoro's voices:
`af_heart` and `am_puck` (American) and `bf_isabella` and `bm_fable` (British).

- **Small.** The int8 model is one 8.5 MB ONNX file for all four voices, against 325 MB for Kokoro.
- **Fast.** It runs about 62x faster than real time on one CPU thread, with no GPU.
- **Close to its teacher.** With `af_heart` it scores 4.40 on UTMOS (Kokoro: 4.52), and its word error rate is within a few tenths of a point of Kokoro's on all four voices.

**New in 1.2:** four voices instead of one, with British pronunciation for the two British voices. See [What changed in 1.2](#what-changed-in-12).

Paradee also runs inside the browser. It is the light voice in the
[Chickadee](https://github.com/sahilmahendrakar/chickadee) Chrome extension, where it needs no WebGPU and no download.

**Paper:** [Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model](paper/Paradee-Mahendrakar-2026.pdf) ([arXiv:2610.06817](https://arxiv.org/abs/2610.06817))

**Model files and audio samples:** [huggingface.co/sahilmahendrakar/Paradee-8M-v1.2](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2) (earlier releases are at [Paradee-8M-v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1) and [Paradee-8M-v1.0](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0))

## Quick start

```bash
pip install git+https://github.com/sahilmahendrakar/paradee
python -m paradee "Paradee is a small voice that runs anywhere." -o hello.wav
python -m paradee "Paradee is a small voice that runs anywhere." --voice bf_isabella -o hello_isabella.wav
```

From Python:

```python
from paradee import Paradee, SAMPLE_RATE
import soundfile as sf

tts = Paradee()   # downloads the 8.5 MB model from the Hugging Face Hub the first time
audio = tts("Paradee is a small voice that runs anywhere.")                     # af_heart
audio = tts("Paradee is a small voice that runs anywhere.", voice="bm_fable")   # or bf_isabella, am_puck
sf.write("hello.wav", audio, SAMPLE_RATE)
```

Text is turned into phonemes with [misaki](https://github.com/hexgrad/misaki), Kokoro's own
grapheme-to-phoneme library, which is what the training data used. The first run also downloads
misaki's English dictionary data. The British voices use misaki's British dictionary, as Kokoro
does. `Paradee(version="1.1")` and `Paradee(version="1.0")` load the earlier releases, which have
`af_heart` only.

## What changed in 1.2

The model is the same size and shape as 1.1. Each half now has a small table with one learned
vector per voice, and the voice you pick chooses the row. Everything else is shared. Both halves
started from 1.1 and were trained further on Kokoro reading 12,000 sentences in each voice.

On the same 200 held-out sentences, with the int8 ONNX file:

| Voice | UTMOS | Kokoro | WER | Kokoro |
|---|---:|---:|---:|---:|
| `af_heart` | 4.40 | 4.52 | 5.8% | 5.7% |
| `bf_isabella` | 4.06 | 4.41 | 5.7% | 5.6% |
| `am_puck` | 4.32 | 4.47 | 5.4% | 6.3% |
| `bm_fable` | 4.18 | 4.32 | 5.7% | 5.7% |

`af_heart` scores the same as in 1.1 (4.39), and the file size and speed are unchanged.
`bf_isabella` is the voice furthest from its teacher.

## What changed in 1.1

Version 1.0 used Kokoro's own audio generator at a smaller width, followed by a filter that fixed
the phase of voiced sound. Version 1.1 replaces both with a new head. The rest of the model (the
text side and the first part of the decoder) is the same.

The new head works at 80 frames a second and never upsamples, which is where the speed comes from.
For voiced sound it builds clean sine waves at the pitch and its multiples, then predicts a smooth
gain and a small phase correction for them. For everything else it shapes real noise, including
noise that pulses in time with the pitch, which is what Kokoro's upper frequencies mostly are.
Because the sine waves are generated from the pitch, their timing is right by construction, and
the separate phase filter is no longer needed.

Both ONNX files, on the same 200 held-out sentences and one CPU thread of an Apple M4 Pro:

| Release | Params | int8 file | Speed | UTMOS | WER |
|---|---:|---:|---:|---:|---:|
| Paradee 1.0 | 8.07M | 9.0 MB | 18x | 4.41 | 5.9% |
| **Paradee 1.1** | **7.88M** | **8.5 MB** | **62x** | **4.39** | **5.9%** |

UTMOS does not measure the buzz. By ear, 1.1 is the cleaner of the two. One measure that does
track it is how much energy sits on the multiples of the pitch compared with halfway between them,
between 2 and 4 kHz. Kokoro measures 3.1 dB, version 1.0 measures 1.3 dB and version 1.1 measures 3.9 dB.

The paper describes version 1.0.

## How it compares

All numbers are on the same 200 held-out sentences, from the paper, which describes version 1.0. Speed is on one CPU thread of
an Apple M4 Pro.

| Model (voice) | Params | File size | Speed | UTMOS | WER |
|---|---:|---:|---:|---:|---:|
| Kokoro-82M, teacher (af_heart) | 81.8M | 325 MB | 7.6x | 4.52 | 5.7% |
| **Paradee 1.0 (af_heart)** | **8.07M** | **8.45 MB** | **25.0x** | **4.41** | **5.7%** |
| Kokoro-7M-Distill (af_msa) | 7.48M | 30.1 MB | 35.5x | 4.18 | 7.4% |
| Piper, en_US-lessac-medium | 15.7M | 63.2 MB | 15.4x | 4.36 | 8.8% |
| KittenTTS nano 0.8 (Bella) | 14.0M | 56.8 MB | 10.7x | 4.01 | 5.5% |

UTMOS is a neural network trained on human ratings. It predicts how natural a clip sounds, on a
scale from 1 to 5. WER is the share of words that Whisper (base) transcribes wrongly.

The paper's speed is for PyTorch. The numbers for the released ONNX files are in the table under
[What changed in 1.1](#what-changed-in-11).

## How it works

Kokoro has two halves. A text side reads phonemes and predicts how long each one lasts, the pitch
and loudness over time, and a feature vector for each phoneme. A decoder turns those into audio.
Paradee is mostly Kokoro's own code at smaller widths, trained half by half against the frozen teacher.

1. **Corpus.** Kokoro reads 12,000 WikiText-103 sentences (23.9 hours of audio), and every
   intermediate value is saved.
2. **Text side (4.23M parameters).** It learns to predict the teacher's durations, pitch, loudness
   and phoneme features directly.
3. **Decoder (3.65M parameters).** It learns to turn the teacher's saved values into the teacher's
   audio, first with spectrogram losses and then with adversarial training. In 1.1 its last part is
   the new head described above, trained the same way with two extra losses that match how tonal
   and how pulsing the teacher's upper frequencies are.
4. **Assembly.** The two halves are joined with no further training, and the weights are stored
   in int8.

Version 1.0 also needed a filter with no parameters that corrected the phase of voiced sound
between 2 and 8 kHz (phase is the timing of each frequency's wave), to remove a slight buzz. In
1.1 the new head does not have that problem and the filter is gone.

The code for every step is in [`training/`](training/). It runs on one laptop.
To rebuild the 1.2 ONNX files from the released PyTorch weights, see
[step 6 in the training README](training/README.md#6-exporting-12-from-the-released-weights).

## Using the ONNX file directly

The model is one ONNX graph that goes from token ids to audio:

- **Inputs:** `input_ids`, int64 `[1, T]`. These are phoneme ids from Kokoro's vocabulary (in
  `config.json` on the Hub), with a `0` pad token at each end, at most 512 in total. The second
  input is `speed`, float32 `[1]`, where 1.0 is normal speed. In 1.2 the third input is `voice`,
  int64 `[1]`, the voice's position in the `voices` list in `config.json` (0 is `af_heart`).
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
