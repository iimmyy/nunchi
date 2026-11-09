# Nunchi

A live Korean to English interpreter that runs on ordinary computers, offline. One small model does the hearing,
the translating and the talking.

- **interpreter mode:** a Korean speech goes in, live English captions and an English voice come out, sent to everyone's
  phone over a Wi-Fi network the computer makes itself. It runs on a plain office PC with no GPU.
- **Conversation mode:** two people talk through a phone, offline (next).

The model is called **Seoul**: LiquidAI's LFM3-Omni-2B with my changes. Design: [docs/design.md](docs/design.md).

## Design rule

Aim for the worst, hope for the best. Everything gets timed on the weakest plausible CPU first (our floor box: a 2014
Intel Haswell server chip, 2 real cores, AVX2 only, 7.5 GB). If it works there, it works everywhere.

## Where it stands (2026-11-09)

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
On imi: brain 34.5 ms, voice generator 18.6 ms, codec 6.2 ms, so 1.35x real time.

**imi** ([imi/](imi/)): an engine written from scratch for Seoul alone. Its own thread team, table layout and math
kernels; ggml only reads the model file. Every piece is checked against an exact float64 reference (a "truth key")
and against the llama.cpp fork.

- Ears: 1.9e-6 from the truth key (the fork: 5.0e-4), 0.03 CPU-seconds per second of sound.
- Tokenizer: the same ids as the fork on 206 test texts (Korean, English and edge cases), every one decoding back.
- Hearing and translation: the same Korean and the same English as the fork on window 1, character for character.
- Live pipeline: while one window is spoken, the next one is heard and translated, and each text step rides on a
  voice frame's brain step (one read of the weights, two conversations). Every output is byte for byte the same as
  running each step alone.

**The whole pipeline on the floor box, 4 threads** (CPU-seconds per second of sermon; under 1.0 keeps up live)

| | llama.cpp fork | imi |
|---|---|---|
| Hearing, window 1 | 0.559 | 0.329 |
| Translation, window 1 | 0.229 | 0.145 |
| Hear, translate and speak, 20 windows, one step at a time | 2.02 | 1.218 |
| Same 20 windows, live (text steps riding the voice) | | **1.035** |

One window in 20 runs away: the voice loses its place in the English and rereads the last sentence for about a
minute. Leaving that window out, the live pipeline is at 0.955. The fix being tested is reading one sentence at a time.

**Wide ears** ([ears/widen_seoul.py](ears/widen_seoul.py)): a side path that lets Seoul hear 8 to 21.5 kHz, trained
on subtitled sermons through real room recordings. Ties stock on a dry feed (9.7 vs 9.6), not yet clearly better in a
room. Still in progress.

## Layout

| Folder | What |
|---|---|
| [docs/](docs/) | The design |
| [seoul/](seoul/) | Model cuts: word table trim, 4-bit output matching, multimodal file converter |
| [ears/](ears/) | Wide-ears training |
| [imi/](imi/) | imi, the engine Seoul runs on |
| [llama.cpp/](llama.cpp/) | Our changes to llama.cpp, as patches |
| [eval/](eval/) | Hearing tests through llama-server, calibration text, plain character-error scoring |
| [memory/](memory/) | Memory research carried over from Little Guy: keeping what was already said |

No model weights, recordings or subtitles are in this repo.

## Who

built by me and claude