# Mixed bits for Seoul: what we measured (November 2026)

The idea: not every number in the brain needs the same precision. Some pieces could keep their answer with fewer
digits, others need more. So hand out bits piece by piece instead of one width for the whole file, and build the
engine around it. We measured it step by step before building anything. Seoul is LFM3-Omni-2B with a trimmed word
table; today's room brain is 4-bit (Q4_0) with its codes re-picked to match the full model's outputs
(`seoul/match.py`) and a 6-bit word table.

## How

- **What each table really sees.** imi records the inputs of every brain table while it hears (12 sermon clips),
  translates (24 passages) and speaks (30 English sentences): `IMI_REC=dir` (core/seoul.cpp), research only.
- **Each piece's error.** `imi-piece-err` (core/piece_err.cpp) measures every piece (8 rows x 256 columns, the
  engine's reading unit) of every table at 2, 3, 4, 5, 6 and 8 bits on those inputs, and per job.
- **Each table's weight in the whole brain.** One table at a time at 2 bits, all others at 4: drift (KL) from the
  16-bit brain on Korean sermon subtitles (`seoul/mixed-bits/analyze_sens.py`).
- **Reading speed.** `imi-kq-bench` (core/kq_bench.cpp) times one input through a table far bigger than the caches,
  for every type, on a 2014 Intel Haswell (the oldest room computer we plan for), with a throwaway 8-row 2-bit reader.
- **Whole brains.** Mixed brains are spliced table by table from finished files (`seoul/mixed-bits/splice.py`, byte
  for byte), then tested like the room's: drift on two sets of sermon subtitles, hearing (85 windows of a held-out
  sermon), translation (20 passages, read by hand), voice (12 held-out sentences spoken and read back).

## What we found

- The idea holds up as a measurement. Piece errors add up inside a table. Single-table harms add up across the
  brain (1.06 against 1.12 for the all-2-bit brain). Tables differ about 20x in how much they matter. The first
  seven layers matter most; layers 15-21 barely notice 2 bits. Inside one table pieces are fairly even, so per-table
  widths do almost as well as per-piece (about 5% apart).
- **More bits do not make Seoul better.** Every layer table at 8 bits drifts 12x less than today's brain (KL 0.006
  vs 0.073). It heard the test sermon at 10.9% characters wrong against 10.8% and translated no better (better in
  5 of 20 passages, worse in 4). Re-matching today's 4-bit brain on all three jobs cut its table errors 3-7x on
  hearing and voice inputs. It also changed nothing: hearing 10.9%, voice perfect in both, translation within the
  same wobble.
- **Fewer bits buy little speed on old processors.** imi's 4-bit reader is limited by math per number as much as by
  memory. A 2-bit table has 58% of the bytes but reads in 80-88% of the time. Handing out bits by reading time
  predicts double the drift for 5% less time.
- So today's 4-bit brain stays. Rounding is not where the translation loses quality. Most wrong English starts as a
  hearing mistake, and hearing is worst where the preacher reads Scripture aloud.
