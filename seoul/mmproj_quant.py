"""Quantize chosen matrices inside a multimodal projector GGUF (mmproj), which llama-quantize does not handle.

  python3 mmproj_quant.py GGUF_PY_DIR IN.gguf OUT.gguf TYPE PREFIX[,PREFIX...]

TYPE: Q8_0 (gguf-py has a numpy quantizer for it). Only 2-D or 3-D (stacked tables) F16/F32 tensors whose name starts with one of the
prefixes and whose row length is a multiple of 32 are converted; everything else (norms, biases, convolutions,
embeddings read by row) is copied unchanged. For Seoul: a.gen = the voice generator (frame generator + codec).
"""
import sys

sys.path.insert(0, sys.argv[1])
import gguf
from gguf.quants import quantize
import numpy as np

src, out, qtype, prefixes = sys.argv[2], sys.argv[3], sys.argv[4], tuple(sys.argv[5].split(","))
QT = {"Q8_0": gguf.GGMLQuantizationType.Q8_0}[qtype]
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(out, r.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=r.endianess)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
datas, n_q, b0, b1 = [], 0, 0, 0
for t in r.tensors:
    d = t.data
    b0 += t.n_bytes
    if (t.name.startswith(prefixes) and len(t.shape) in (2, 3) and t.name.endswith(".weight")
            and t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)
            and int(t.shape[0]) % 32 == 0 and int(t.shape[1]) >= 32):
        q = quantize(np.asarray(d, dtype=np.float32), QT)
        w.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, QT)
        datas.append(q)
        n_q += 1
        b1 += q.nbytes
    else:
        w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
        datas.append(d)
        b1 += t.n_bytes
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for d in datas:
    w.write_tensor_data(d, tensor_endianess=r.endianess)
w.close()
print(f"{out}: {n_q} matrices to {qtype}; {b0 / 1e6:.0f} MB -> {b1 / 1e6:.0f} MB")
