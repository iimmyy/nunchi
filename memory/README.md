# memory

The memory research behind how Seoul keeps what was already said, carried over from Little Guy's memory work
(2026-11-08). Each script runs from this folder on its own.

| File | What it does |
|---|---|
| `shrinktest.py` | Tests ways of keeping a chat smaller (fresh words, bricks, surprise cuts, packed bricks, the model's own summaries, the sleep design) on LongMemEval's oracle split; LoCoMo is the practice bank |
| `sae.py` | Finds the pieces ("bricks") inside the model at chosen floors (`--layers`) |
| `rung0.py`, `moments.py` | The first rung of the memory tests and the moments they use (shared helpers) |
| `ogg.py` | Squeezes stored keys and values: a hearing model of what look-ups pick up, a rotation into the directions that matter, bits spent where they are heard |
| `marks.py` | Writes the calendar date right after relative times in what the user said ("yesterday [= 2023/05/20 (Sat)]") |
| `remind.py` | Tests whether the model's own surprise notices a changed fact in a later chat |
