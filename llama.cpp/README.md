# Our llama.cpp changes

Base: llama.cpp commit `604cc43` (b11963), which runs LFM3-Omni (hearing, voice, `/v1/realtime`).

```bash
git -C llama.cpp checkout 604cc43
git -C llama.cpp apply ../nunchi/llama.cpp/lfm3-voice.patch ../nunchi/llama.cpp/imatrix-act-dump.patch
```

| Patch | What |
|---|---|
| `lfm3-voice.patch` | **Word table skip:** `llama_set_skip_logits()` leaves out the LM head when only the hidden state is needed. The LFM3 voice loop uses it on audio frames, and `llama-tts` samples a word only when the pipeline reads one (`mtmd_helper_gen_audio_needs_sampled`). Audio is byte-identical. Also research switches: `LG_TTS_STAGES=1` prints time per stage per frame, `LG_GEN_DUMP=file` writes every voice-generator pass (the answer key for imi.cpp), `LG_GEN_SCHEDULE=6,6` changes how many codebooks each pass reveals (reference: 4,4,4) |
| `imatrix-act-dump.patch` | `LG_ACT_DUMP=<dir>` in `llama-imatrix` saves the real input rows of chosen matrices, for 4-bit output matching (`seoul/match.py`). `LG_ACT_WANT`, `LG_ACT_MAX`, `LG_ACT_EVERY`, `LG_ACT_LAYERS` pick what is saved |

llama.cpp is MIT licensed (copyright the ggml authors); these patches are derived from it.
