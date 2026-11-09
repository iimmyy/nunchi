# Nunchi: live Korean and English interpreter (design, 2026-11-08)

Status: draft for Immy's review. Nothing here is built yet except the Seoul ear experiments.

## 1. What it is

Nunchi is one app with one model and two modes. The model is **Seoul** (LiquidAI LFM3-Omni-2B with our changes).
Seoul hears, sees and talks, which is why it was picked: one small model can run a live conversation.

| Mode | What it does | Runs on | When |
|---|---|---|---|
| **Sermon** (one way) | Korean sermon in, live English captions out to everyone's phone | The church's Windows computer (a regular Dell, CPU only) | First, this Sunday (2026-11-15) |
| **Conversation** (two way) | Immy and a Korean speaker talk through his phone | Immy's iPhone 18 Pro, fully offline | Right after, as a test build; for the Korea trip in February 2027 |

Immy's decisions this stands on:
- Instant beats delayed but perfect. Captions must feel live, not 15 seconds behind.
- Sermon mode runs on the church's own Dell. No laptop to bring, no cloud, no home server.
- Conversation mode runs on the phone from day one, offline.
- Seoul is made smaller: Korean and English only, everything else stripped. Eyes stay (conversation mode), not used
  in sermon mode.
- No prompt tricks, no thinking (it adds delay).
- The room-echo training stays as the fallback for when there is no direct sound feed.

## 2. Sermon mode

### How it fits together

```
church mixer --(their existing audio input)--> Dell
Dell: voice detector -> Seoul (llama.cpp, CPU) -> Korean text, live
                                               -> English per phrase
Dell: makes its own Wi-Fi network (the church has no Wi-Fi) + small local web server -> captions page (+ QR code)
phones in the room -> open the page -> read along and listen (English voice in earbuds)
projector (optional) -> the same page
Immy's website -> a landing page with instructions and the download; links to the local page
```

### Pieces

| Piece | Job | Notes |
|---|---|---|
| Audio in | Read the sermon from the Dell's sound input | The church already has the mixer feed on that computer |
| Voice detector | Cut at short pauses (every few seconds), wake Seoul only when someone speaks | A tiny model or an energy rule; CPU is precious |
| Hearing | Korean text as it is spoken | Seoul's ears have a built-in streaming mode; llama.cpp's live transcription must be checked for LFM3-Omni (see risks) |
| Translating | English for each finished phrase | Text only, short, so fast; recent phrases go in as earlier chat turns (the format that never echoed in our tests) |
| Network | The Dell makes the Wi-Fi network phones join (there is no church Wi-Fi) | Windows Mobile Hotspot needs a Wi-Fi adapter in the Dell (a USB one if it has none) and may refuse to start with no internet to share; checked on the Dell early in the week |
| Captions server | Serve the page and push new lines to every phone as they come | Local network only. Browsers block a public website from talking to a local computer, so the live page comes from the Dell itself |
| Captions page | Big readable English, Korean optional, auto-scroll | Works on any phone browser, nothing to install |
| Voice | Seoul speaks the English in his own voice, in step with the captions (Immy 11-08: "at this speed we do both") | Seoul's interleaved text-and-audio mode: a piece of text, then its audio, in one stream, so they stay in sync. The page plays it for anyone with earbuds. Risk: the voice's speed on the Dell's CPU (3x real time on the Mac GPU, not yet measured on a CPU) |
| Landing page | On Immy's site: what it is, how to connect, the download | |

### Speed budget (the target, to be measured)

Korean puts the verb last, so a phrase's English cannot come before the phrase ends. The target is as close to that
limit as the Dell allows:
- Korean text on screen: about 1 s behind the speaker.
- English for a phrase: 1 to 3 s after the phrase ends.

How it gets checked before Sunday: llama.cpp on Immy's Mac, CPU only, capped to a basic desktop's threads, timed per
phrase on the held-out test sermon. Then on the Dell itself as soon as we can reach it.

### Ears

Both modes use the wide ears (8 to 21 kHz). Immy: when we later fine-tune Seoul and cut him down for size and cost,
the bigger ears are what keeps the hearing good. They cost about one million extra weights (2 to 4 MB next to Seoul's
1.3 GB) and very little compute. That means the wide-ears side path has to run inside llama.cpp (our fork) before
Sunday, checked against the PyTorch version the same way the Gemma port was. On a dry mixer feed stock Seoul was
already close to word for word (first 20 s of the held-out sermon), so the wide ears must at least match that there.

## 3. Conversation mode

| Part | Design |
|---|---|
| Starting | One switch starts a conversation; it stays on until switched off |
| Quiet | When nobody talks, only the voice detector runs; Seoul sleeps (battery) |
| Direction | Seoul tells Korean from English and translates the right way, no tapping |
| Sound out | Their Korean comes to Immy in English in his earbuds; his English goes to them in Korean from the phone speaker |
| Eyes | Point the camera at a sign or a menu and get it in English |
| Runtime | llama.cpp built into the iPhone app (its prebuilt iOS framework, Metal), Seoul as a GGUF file |
| Wide ears | Ported into our llama.cpp fork the same way the Gemma wide ears were (fork patch 0048) |
| Testing | Immy and Korean-speaking friends, in person |

## 4. Seoul itself

| Change | Why | How it is checked |
|---|---|---|
| Wide ears (8 to 21 kHz side path, trained on subtitled sermons, half through the church's own clap recordings), in both modes | Hear consonants and room sound better, and keep hearing good when Seoul is later fine-tuned and cut down | Against stock Seoul and an 8 kHz control, scored against hand subtitles on a held-out sermon, plus my reading |
| Korean and English vocabulary only | The 128,000-word table is read in full for every word Seoul writes (about 215 MB per word); keeping the Korean, English, number and punctuation pieces makes each word about 15% cheaper on a CPU | Same Korean and English answers as full Seoul on the test sermon |
| 4-bit with our pipeline (importance-searched scales, every table's type checked, each step against 16-bit) | A CPU reads about 3x fewer bytes per word | Drift against 16-bit Seoul, table by table. Seoul has no QAT grid, so the scale search carries over from the E4B work, not the grid alignment |
| ogg (KV squeeze) on the running notes | The recent phrases Seoul keeps as already-read context stay cheap on the Dell and the phone | Held the memory-test score to 8x; checked here on the test sermon with and without |
| Compressor (Immy's sidechain design, c5, built on LFM3, Seoul's own brain family) | On LFM3 it cut thinking and raised right answers; here it is tested on the failure we saw in translation: rambling and loops | Ported, then loops and accuracy with and without it on the test sermon and the pew recording. Kept only if it helps |
| Cutting inside the brain (neurons and whole layers), then healing | **Last resort** (Immy). Only if Seoul still cannot run fast enough on the Dell's CPU after everything above | Done at the very end, if at all, on the wide-ears Seoul; every cut checked against the uncut Seoul on the test sermon |

Loading size is not a goal (the Dell has at least 8 GB); speed per word is.

## 5. Testing

- Speed: per phrase, on the Mac (capped) and on the Dell.
- Accuracy: character error rate against the hand-made Korean subtitles of the held-out sermon (plain counting, no AI
  judge), the English read by hand against the hand-made English subtitles, and every window read by me.
- The pew recording stays the worst-case check.
- Conversation mode: Immy with Korean friends.

## 6. Risks and open questions

| Risk | What we do |
|---|---|
| The Dell is slower than estimated (unknown CPU and RAM) | Get its specs or time it on site early in the week; the smaller Seoul is the main lever |
| llama.cpp's live transcription may only cover LFM3-Audio, not LFM3-Omni | Check first thing; if missing, cut at pauses and run normal ASR per phrase (still a few seconds per phrase) |
| A desktop Dell may have no Wi-Fi card, or Windows may not start a hotspot with no internet | A USB Wi-Fi adapter; if Windows still refuses, a small travel router plugged into the Dell makes the network on its own |
| The wide ears are not running in llama.cpp by Sunday | They are the first build job; checked against PyTorch before anything uses them |
| Installing on the church's computer needs their OK | Immy explains it to them (ideally through Nunchi) |
| Windows audio input setup | Use whatever input the church already records from |

## 7. Not in this version

Video, translation into languages other than English, accounts or
cloud anything.
