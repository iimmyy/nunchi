"""Write a Seoul GGUF with a smaller word table: only the pieces in a keep list (from seoul_vocab.py keep).

  python3 seoul_trim.py GGUF_PY_DIR IN.gguf KEEP.json OUT.gguf

Seoul's output layer is the word table itself (tied), so every word Seoul writes reads the whole table. Dropping
rows makes each word cheaper and the model can no longer pick a dropped piece. Pieces are renumbered in their old
order; the special-token ids, the merge rules and the table rows follow. Everything else is copied unchanged.
"""
import json, sys

sys.path.insert(0, sys.argv[1])
import gguf
import numpy as np

src, keep_path, out = sys.argv[2:5]
r = gguf.GGUFReader(src)
F = r.fields
tokens = F["tokenizer.ggml.tokens"].contents()
ttype = F["tokenizer.ggml.token_type"].contents()
merges = F["tokenizer.ggml.merges"].contents()
keep = json.load(open(keep_path))["keep"]
assert keep == sorted(set(keep)) and keep[-1] < len(tokens)
new_id = {old: new for new, old in enumerate(keep)}
kept_tok = [tokens[i] for i in keep]
kept_set = set(kept_tok)
kept_merges = [m for m in merges if (lambda a, b: a in kept_set and b in kept_set and a + b in kept_set)(*m.split(" ", 1))]

ID_KEYS = [k for k in F if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")]
for k in ID_KEYS:
    old = int(F[k].contents())
    assert old in new_id, f"{k}={old} ({tokens[old]!r}) is not in the keep list"

w = gguf.GGUFWriter(out, F[gguf.Keys.General.ARCHITECTURE].contents(), endianess=r.endianess)
for field in F.values():
    name = field.name
    if name == gguf.Keys.General.ARCHITECTURE or name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if name == "tokenizer.ggml.tokens":
        val = kept_tok
    elif name == "tokenizer.ggml.token_type":
        val = [ttype[i] for i in keep]
    elif name == "tokenizer.ggml.merges":
        val = kept_merges
    elif name in ID_KEYS:
        val = new_id[int(val)]
    w.add_key_value(name, val, val_type, sub_type=sub_type)

datas = []
for t in r.tensors:
    d = t.data
    if t.name == "token_embd.weight":
        assert d.shape[0] == len(tokens), d.shape
        d = np.ascontiguousarray(d[keep])
        shape = [len(keep)] + list(d.shape[1:])
        w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
    else:
        w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
    datas.append(d)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for d in datas:
    w.write_tensor_data(d, tensor_endianess=r.endianess)
w.close()
print(f"{out}: {len(keep)} of {len(tokens)} pieces, {len(kept_merges)} of {len(merges)} merges; "
      + ", ".join(f"{k.split('.')[-1]} {int(F[k].contents())}->{new_id[int(F[k].contents())]}" for k in ID_KEYS))
