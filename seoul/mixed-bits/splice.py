"""Mixed-bits brains (2026-11-10): one GGUF whose tables come from different files, table by table.

  python3 splice.py OUT.gguf BASE.gguf CHOICE.tsv LABEL=FILE [LABEL=FILE ...]

BASE gives the file's settings and every tensor that CHOICE.tsv does not name (here: the room's brain, matched 4-bit
with the 6-bit word table). CHOICE.tsv: one "tensor<TAB>label" per line; that tensor is copied byte for byte from the
file given for that label (matched 3-bit, matched 2-bit, 8-bit, ...). Every source must be the same brain (same names
and shapes). Nothing is re-rounded here: each table is exactly as its source file has it."""
import os, sys
from pathlib import Path

if os.environ.get("GGUF_PY"):   # the gguf-py folder of our llama.cpp fork, unless gguf is installed
    sys.path.insert(0, os.environ["GGUF_PY"])
import gguf


def main():
    out, base_p, choice_p = sys.argv[1:4]
    src = dict(a.split("=", 1) for a in sys.argv[4:])
    base = gguf.GGUFReader(base_p)
    readers = {lab: {t.name: t for t in gguf.GGUFReader(p).tensors} for lab, p in src.items()}
    pick = dict(l.rstrip("\n").split("\t")[:2] for l in open(choice_p) if l.strip())
    names = {t.name for t in base.tensors}
    for n, lab in pick.items():
        if n not in names: raise SystemExit(f"{n}: not in the base file")
        if lab not in readers: raise SystemExit(f"{n}: no file given for label {lab}")
        a, b = readers[lab][n], next(t for t in base.tensors if t.name == n)
        if list(a.shape) != list(b.shape): raise SystemExit(f"{n}: shape {list(a.shape)} vs base {list(b.shape)}")
    w = gguf.GGUFWriter(out, base.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=base.endianess)
    for f in base.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    chosen = [readers[pick[t.name]][t.name] if t.name in pick else t for t in base.tensors]
    for t in chosen:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in chosen:
        w.write_tensor_data(t.data, tensor_endianess=base.endianess)
    w.close()
    count = {}
    for t in chosen: count[t.tensor_type.name] = count.get(t.tensor_type.name, 0) + int(t.n_elements)
    tot = sum(count.values())
    print(f"wrote {out}: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in sorted(count.items())) +
          f" of {tot} numbers, {os.path.getsize(out)} bytes")


if __name__ == "__main__":
    main()
