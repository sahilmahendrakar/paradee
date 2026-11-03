# Training Paradee

This folder has the code that produced Paradee v1.0 and v1.1. Steps 1 to 4 are v1.0, and step 5
adds the v1.1 head on top of them. Step 6 exports 1.2 from its released weights. Every run fits
on one laptop: the original ran on a MacBook Pro with an Apple M4 Pro and 24 GB of memory, using
PyTorch's MPS backend. Run every command from this folder. The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
cd training
uv sync
```

The scripts write to `data/`, `checkpoints/`, `models/` and `out/`.

## 1. Teacher corpus

`data/wikitext_sents.json` holds the 12,000 WikiText-103 sentences that were used. The first 200
are held out for evaluation. Kokoro reads each sentence with the `af_heart` voice, and the script
saves the phonemes, durations, pitch, loudness, phoneme features and audio in shards of 500
sentences (6.9 GB in total).

```bash
uv run python scripts/gen_teacher.py 12000
uv run python scripts/regen_audio.py
```

`regen_audio.py` re-renders every waveform on the CPU with a fixed seed per sentence. Kokoro's
output differs slightly between CPU and GPU, and the decoder is trained to match the CPU version.

## 2. Text side (4.23M parameters)

This step learns to predict the teacher's durations, pitch, loudness and phoneme features. It runs
for 8,000 steps at batch size 32.

```bash
uv run python scripts/train_text.py s+mlp --steps 8000
```

## 3. Decoder (3.85M parameters)

The decoder is trained in three runs. Each run starts from the one before it.

```bash
# stage one: spectrogram losses only, 50,000 steps at batch size 16
uv run python scripts/train_decoder.py A
# stage two: adversarial training, spectrogram losses weighted 10x, then 3x
uv run python scripts/train_decoder.py A --gan --mel-weight 10 --init-from checkpoints/dec_A/last.pt --steps 5000 --bs 8 --tag _gan10
uv run python scripts/train_decoder.py A --gan --mel-weight 3 --init-from checkpoints/dec_A_gan10/last.pt --steps 5000 --bs 8 --shards 8 --tag _gan3
```

Every trainer can be resumed: running the same command again continues from that run's `last.pt`.
Run at most two trainers at once on a 24 GB machine.

## 4. Assembly, phase filter and export

This step joins the two halves with no further training. It adds the phase-locking filter (see
`scripts/phase_lock.py`) and writes one ONNX graph in fp32 plus a 9 MB int8 copy. It also checks
that the ONNX filter matches the PyTorch one.

```bash
uv run python scripts/export_paradee.py
```

Kokoro ships an ONNX-friendly STFT (`kokoro.custom_stft.CustomSTFT`). It does not match
`torch.stft`, which the decoder was trained with, and it costs about 0.5 UTMOS. The export uses
its own exact version (`ConvSTFT`) instead.

## 5. The v1.1 head

Version 1.1 keeps the text side and the first part of decoder A, and replaces the audio generator
and the phase filter with a head that works at 80 frames a second (`scripts/vhead.py`, head `v2`).
The trainer starts from the stage-two decoder of step 3.

```bash
X=(--head v2 --fine-stft --harm-weight 2 --mod-weight 5 --shards 999 --utmos)
# stage one: spectrogram losses plus the two statistic losses, all 24 shards
uv run python scripts/train_vhead.py $X --steps 40000 --eval-every 10000 --tag 2_full
# stage two: adversarial training, spectrogram losses weighted 3x, batch size 8
uv run python scripts/train_vhead.py $X --gan --mel-weight 3 --bs 8 --init-from checkpoints/vh2_full/last.pt --steps 15000 --tag 2_fullgan
# one ONNX graph, fp32 and int8
uv run python scripts/export_paradee_v1_1.py checkpoints/vh2_fullgan/last.pt
```

`--harm-weight` matches how much of the teacher's energy sits on multiples of the pitch, per band.
`--mod-weight` matches how the loudness of the upper bands pulses with the pitch. Spectrogram
losses alone leave the upper harmonics too clean, which sounds buzzy.

The released head did not start stage one from scratch. It was first developed on 8 shards, and
those runs were carried over: 20,000 steps with spectrogram losses, about 13,000 more as the two
statistic losses were added, and 8,000 after the short-window noise path was added. The 40,000
steps on all the data and the 15,000 adversarial steps then followed as above. Starting stage one
from scratch with the commands above has not been tested end to end.

```bash
uv run python scripts/vh_ab.py checkpoints/vh2_fullgan/last.pt 200 --wer   # v1.0 against the new head, PyTorch
uv run python scripts/onnx_eval.py 200 v1.1=models/paradee_v1.1/paradee_int8.onnx   # a released-style ONNX file
```

## 6. Exporting 1.2 from the released weights

`scripts/export_paradee_v1_2.py` writes the 1.2 ONNX graph (four voices, a third input `voice`)
in fp32 and int8 from the PyTorch weights on the Hub. It needs no teacher data or training.

```bash
uv run hf download sahilmahendrakar/Paradee-8M-v1.2 config.json pytorch/text_side.pt pytorch/decoder.pt --revision v1.2 --local-dir hf
uv run python scripts/export_paradee_v1_2.py hf/pytorch/decoder.pt --text hf/pytorch/text_side.pt --config hf/config.json
```

It first checks that the export graph matches the training code for every voice, then writes
`models/paradee_v1.2/paradee.onnx` and `paradee_int8.onnx` and prints the speed of each on one CPU
thread. It takes about a minute on a CPU. The released files were exported with torch 2.14.1, and
with that version both come out byte for byte the same as the ones on the Hub (int8 sha256
`a6989796...6c97da1`). With the locked torch 2.14.0 the weights and ops are the same and only the
file header differs. To match exactly, run `uv pip install torch==2.14.1` first and add
`--no-sync` to `uv run`. The four-voice style tables live in `student.py` and `vhead.py`
(`n_voices`), and checkpoints with one voice still load as before.

## Evaluation

```bash
uv run python scripts/eval_full.py s+mlp A --dec-tag _gan3   # student against teacher, with listening WAVs
uv run python scripts/rtf.py s+mlp A --dec-tag _gan3   # speed on one CPU thread
uv run python scripts/utmos.py out/some_dir            # UTMOS (utmos22_strong)
uv run python scripts/wer.py out/some_dir              # Whisper (base) WER against a .txt next to each WAV
```

## Other files

| Script | What it does |
|---|---|
| `common.py` | Loads the Kokoro teacher, plus shared metrics (DTW log-mel distance) |
| `student.py` | The text-side student: Kokoro's text modules at smaller widths, with a learned constant voice |
| `student_decoder.py` | The decoder student: Kokoro's decoder at configurable widths (preset `A` is Paradee) |
| `discriminators.py` | The multi-period and multi-resolution discriminators used in stage two |
| `vhead.py` | The v1.1 head and the decoder that uses it |
| `quantize_student.py` | Weight-only int8 in PyTorch, with its effect on quality |
