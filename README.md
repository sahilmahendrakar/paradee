# Paradee

Paradee is a small text-to-speech model. It has 7.88M parameters and is distilled from
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), and it speaks eight of Kokoro's voices
in four languages:

| Language | Voices |
|---|---|
| English | `af_heart`, `am_puck` (American), `bf_isabella`, `bm_fable` (British) |
| Hindi | `hf_alpha`, `hm_omega` |
| French | `ff_siwis` |
| Spanish | `em_alex` |

- **Small.** The int8 model is one 8.5 MB ONNX file for all eight voices, against 325 MB for Kokoro.
- **Fast.** It runs about 62x faster than real time on one CPU thread, with no GPU.
- **Close to its teacher.** In English, French and Spanish, Whisper makes about as many mistakes on Paradee as on Kokoro, and in Hindi one or two more words in a hundred. See [What changed in 1.3](#what-changed-in-13).

**New in 1.3:** Hindi, French and Spanish, in the same file and at the same speed.

Paradee also runs inside the browser. It is the light voice in the
[Chickadee](https://github.com/sahilmahendrakar/chickadee) Chrome extension, where it needs no WebGPU and no download.

**Paper:** [Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model](paper/Paradee-Mahendrakar-2026.pdf) ([arXiv:2610.06817](https://arxiv.org/abs/2610.06817))

**Model files and audio samples:** [huggingface.co/sahilmahendrakar/Paradee-8M-v1.3](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.3) (earlier releases are at [Paradee-8M-v1.2](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2), [Paradee-8M-v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1) and [Paradee-8M-v1.0](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0))

## Quick start

```bash
pip install git+https://github.com/sahilmahendrakar/paradee
python -m paradee "Paradee is a small voice that runs anywhere." -o hello.wav
python -m paradee "Paradee is a small voice that runs anywhere." --voice bf_isabella -o hello_isabella.wav
python -m paradee "La biblioteca abre a las nueve." --voice em_alex -o hola.wav
```

From Python:

```python
from paradee import Paradee, SAMPLE_RATE
import soundfile as sf

tts = Paradee()   # downloads the 8.5 MB model from the Hugging Face Hub the first time
audio = tts("Paradee is a small voice that runs anywhere.")                     # af_heart
audio = tts("Paradee is a small voice that runs anywhere.", voice="bm_fable")   # or bf_isabella, am_puck
sf.write("hello.wav", audio, SAMPLE_RATE)
sf.write("namaste.wav", tts("आज मौसम बहुत अच्छा है।", voice="hf_alpha"), SAMPLE_RATE)        # Hindi
sf.write("bonjour.wav", tts("Il fait très beau aujourd'hui.", voice="ff_siwis"), SAMPLE_RATE)  # French
```

`with_timestamps` also says when each word is spoken, for captions or for highlighting the word
being read. The times come from the durations the audio was made from:

```python
audio, words = tts.with_timestamps("Paradee is a small voice that runs anywhere.")
# words: [("Paradee", 0.25, 0.84), ("is", 0.84, 0.96), ("a", 0.96, 1.06), ...]: (text, start, end) in seconds
```

The voice sets the language, and each sentence is spoken in one language. English is turned into
phonemes with [misaki](https://github.com/hexgrad/misaki), Kokoro's own grapheme-to-phoneme library
(its British dictionary for the British voices). The first English run also downloads misaki's
dictionary data. Hindi, French and Spanish, and the English words misaki has no entry for, go
through Paradee's own small G2P model, described in [Phonemes without eSpeak](#phonemes-without-espeak). `Paradee(version="1.2")` loads the
previous release with the four English voices, and `Paradee(version="1.1")` and
`Paradee(version="1.0")` load the earlier ones, which have `af_heart` only.

## What changed in 1.3

The model is the same size and shape as 1.2. The table of voice vectors in each half has eight rows
instead of four. The English voices keep their numbers (0 to 3), and the new ones are 4 to 7.
Paradee already used Kokoro's full phoneme vocabulary, which covers all of Kokoro's languages, so
nothing else changed. Both halves started from 1.2 and were trained further on Kokoro reading
Wikipedia sentences in each new voice, together with the English voices.

With the int8 ONNX file, on 200 held-out sentences per voice, Kokoro in brackets:

| Voice | UTMOS | NISQA | WER | CER |
|---|---:|---:|---:|---:|
| `af_heart` | 4.38 (4.52) | 4.93 (5.00) | 2.7% (3.0%) | 0.8% (1.0%) |
| `bf_isabella` | 4.07 (4.41) | 5.08 (5.12) | 2.9% (2.9%) | 1.0% (0.9%) |
| `am_puck` | 4.35 (4.47) | 5.01 (5.04) | 2.8% (3.0%) | 0.9% (0.9%) |
| `bm_fable` | 4.17 (4.32) | 4.96 (5.00) | 2.7% (3.0%) | 0.9% (0.9%) |
| `hf_alpha` | 3.99 (4.29) | 3.98 (4.37) | 21.7% (20.7%) | 6.4% (6.0%) |
| `hm_omega` | 3.82 (4.09) | 4.57 (4.73) | 21.6% (19.8%) | 6.7% (6.1%) |
| `ff_siwis` | 3.45 (3.65) | 4.39 (4.65) | 6.8% (6.8%) | 2.2% (2.2%) |
| `em_alex` | 3.14 (3.34) | 4.72 (4.98) | 5.0% (5.1%) | 1.5% (1.5%) |

WER and CER are the share of words and of characters that Whisper large-v3 transcribes wrongly, in
the voice's own language. NISQA, like UTMOS, predicts how clean a clip sounds. Both were trained
mostly on English ratings, so for Hindi, French and Spanish their scores only mean something next
to Kokoro's. Hindi WER is high for Kokoro too, partly because many Hindi words have more than one
accepted spelling. The English voices score the same as in 1.2, and the file size and speed are
unchanged.

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
[step 6 in the training README](training/README.md#6-exporting-12-from-the-released-weights). The 1.3
export is the same script with eight voices (`--voices af_heart,bf_isabella,am_puck,bm_fable,hf_alpha,hm_omega,ff_siwis,em_alex`).

## Using the ONNX file directly

The model is one ONNX graph that goes from token ids to audio:

- **Inputs:** `input_ids`, int64 `[1, T]`. These are phoneme ids from Kokoro's vocabulary (in
  `config.json` on the Hub), with a `0` pad token at each end, at most 512 in total. The second
  input is `speed`, float32 `[1]`, where 1.0 is normal speed. Since 1.2 the third input is `voice`,
  int64 `[1]`, the voice's position in the `voices` list in `config.json` (0 is `af_heart`).
- **Output:** `waveform`, float32 `[1, samples]` at 24 kHz.
- **Durations:** the graph also computes each token's length in frames of 600 samples, rounded,
  as the tensor `/Clip_output_0`, float32 `[1, T]`, pads included. The frames add up to the
  waveform's length exactly. It is not a graph output in the released files; `paradee/tts.py`
  adds it as one when it loads the file, and `with_timestamps` turns it into word times.

[`web/misaki.js`](web/misaki.js) is for the browser. Browser phonemizers such as
[kokoro-js](https://github.com/hexgrad/kokoro/tree/main/kokoro.js) use eSpeak NG, which writes
some sounds differently from misaki. Paradee only learned misaki's spelling. Without this
conversion Whisper mishears about 31% of words, and with it about 2%. It is for the English voices
only. For Hindi, French and Spanish the model learned eSpeak's own output, with the few symbol
renames in misaki's `EspeakG2P`.

## Phonemes without eSpeak

Kokoro, and the data Paradee was trained on, get the phonemes for Hindi, French and Spanish from
[eSpeak NG](https://github.com/espeak-ng/espeak-ng), and misaki also hands it the English words it
has no entry for, mostly names. eSpeak NG is GPL-3.0, so an app that bundles it has to be GPL too.
Since 1.3.1 the package doesn't use it. [`paradee/g2p.py`](paradee/g2p.py) gives the same
phonemes from three small parts in `paradee/data/`, all Apache 2.0:

- a 0.76M-parameter model (0.9 MB int8 ONNX) that reads one word, plus the last letter of the word
  before it and the first letter of the word after it, and writes the word's phonemes. The
  neighbouring letters are enough for French liaison ("les amis"), the Spanish b, d and g that
  soften after a vowel, and the stress eSpeak puts at the end of a phrase.
- word lists: the words the model gets wrong, and the 3,000 most common words of each language.
- a table of number pieces per language, which numbers are put together from.

The model and lists were learned from eSpeak's output on 400k Wikipedia sentences per language.
On held-out sentences from other articles they differ from eSpeak in 0.80% (Hindi), 0.95% (French)
and 0.59% (Spanish) of phoneme characters, mostly a moved stress mark. For the English words misaki
sends on, they differ in 1.6% (American) and 1.7% (British). On 200 sentences per voice, Whisper's
word error rate on Paradee is within 0.3 points of the eSpeak phonemes for every voice.

`Paradee(g2p="espeak")` (or `--g2p espeak`) uses eSpeak NG as before. It needs
`pip install "paradee[espeak]"`.

## License

Apache 2.0, the same as Kokoro. The sentences in `training/data/` come from WikiText-103 and are
under CC BY-SA 3.0.

## Acknowledgements

Paradee is distilled from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad,
which builds on [StyleTTS 2](https://github.com/yl4579/StyleTTS2) and
[iSTFTNet](https://arxiv.org/abs/2203.02395). Phonemes come from
[misaki](https://github.com/hexgrad/misaki), and Paradee's own G2P was learned from
[eSpeak NG](https://github.com/espeak-ng/espeak-ng)'s output.

## Citation

```bibtex
@misc{mahendrakar2026paradee,
  title  = {Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model},
  author = {Mahendrakar, Sahil},
  year   = {2026},
  url    = {https://github.com/sahilmahendrakar/paradee}
}
```
