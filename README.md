# Nunchi

A live Korean to English interpreter that runs on ordinary computers, offline. One small model does the hearing,
the translating and the talking.

- **Sermon mode:** a Korean sermon goes in, live English captions and an English voice come out, sent to everyone's
  phone over a Wi-Fi network the computer makes itself. It runs on a plain office PC with no GPU.
- **Conversation mode:** two people talk through a phone, offline (next).

The model is **Seoul**: LiquidAI's LFM3-Omni-2B with our changes. Design: [docs/design.md](docs/design.md).

## Design rule

Aim for the worst, hope for the best. Everything gets timed on the weakest plausible CPU first (our floor box: a 2014
Intel Haswell server chip, 2 real cores, AVX2 only, 7.5 GB). If it works there, it works everywhere.

## Where it stands (2026-11-08)

**Seoul made smaller (cut on a Mac, checked against 16-bit at every step)**

| Version | Size | Mean KLD vs 16-bit | Same top token | Dry hearing error (speech) |
|---|---|---|---|---|
| 16-bit | 3.47 GB | 0 | 100% | 9.5% |
| Stock Q4_K_M | 1.10 GB | 0.060 | 87.9% | 9.8% |
| Stock Q4_0 | 1.05 GB | 0.109 | 82.5% | 10.0% |
| Output-matched Q4_0 | 1.05 GB | 0.060 | 87.1% | 9.6% |
| Output-matched Q4_0 + word table trim | 0.999 GB | n/a (vocab differs) | n/a | 9.5% |

- Word table trim ([seoul/seoul_vocab.py](seoul/seoul_vocab.py), [seoul/seoul_trim.py](seoul/seoul_trim.py)): keeps
  Hangul, Latin, Greek, Hebrew, digits, punctuation, all bytes and the pieces that cut across a Korean character,
  closed under the merge rules. 97,496 of 128,000 rows. Tokenization identical on 688,994 tokens of sermon subtitles.
- 4-bit output matching ([seoul/match.py](seoul/match.py)): stock Q4_0 block scales kept, 4-bit codes re-picked
  column by column so each layer's output on Seoul's real inputs matches 16-bit (GPTQ-style error compensation on a
  fixed grid). The result is a normal Q4_0 file any llama.cpp can load. Whole-model held-out error 170.9 to 73.7.
- Multimodal file converter ([seoul/mmproj_quant.py](seoul/mmproj_quant.py)): llama-quantize does not touch the
  ears/voice file, so we wrote our own. Voice generator to 8-bit: 950 MB to 791 MB.

**Speed on CPU, 4 threads**

| | Floor box (2014) | Typical desktop |
|---|---|---|
| 16-bit Seoul, 30 s of sermon | | 31.5 s |
| 1 GB Seoul, 30 s of sermon | 16.2 s | 14.9 s |
| English voice, 16-bit | | 0.31x real time |
| English voice, 1 GB Seoul | 0.62x | 0.64x |
| English voice, 1 GB Seoul + 8-bit generator + word table skip | **0.75x** | |

One voice frame on the floor box (budget 80 ms for real time): brain 45 ms, voice generator 40 ms, codec 8 ms.

**imi.cpp** ([imi/](imi/)): our own engine, built for Seoul alone on top of ggml. First piece, the voice generator:
scores bit-identical to llama.cpp on the Mac CPU (345 of 345 passes), and on the floor box 40.7 to 32.5 ms per frame
with the same math kernels, from structure alone (input projection once per frame, only the 4 codebooks each pass
picks are scored).

**Wide ears** ([ears/widen_seoul.py](ears/widen_seoul.py)): a side path that lets Seoul hear 8 to 21.5 kHz, trained
on subtitled sermons through real room recordings. Ties stock on a dry feed (9.7 vs 9.6), not yet clearly better in a
room. Still in progress.

## Layout

| Folder | What |
|---|---|
| [docs/](docs/) | The design |
| [seoul/](seoul/) | Model cuts: word table trim, 4-bit output matching, multimodal file converter |
| [ears/](ears/) | Wide-ears training |
| [imi/](imi/) | imi.cpp, our engine |
| [llama.cpp/](llama.cpp/) | Our changes to llama.cpp, as patches |
| [eval/](eval/) | Hearing tests through llama-server, calibration text, plain character-error scoring |

No model weights, recordings or subtitles are in this repo.

## Who

Immy and Claude (Anthropic). Immy's designs: the wide ears, the floor-box rule, the English voice in sermon mode, and
building our own engine. Claude writes the code.
