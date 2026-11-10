"""Export Granite-4.0-H-350M (+ merged LoRA) to ONNX by re-using the onnx-community graph and swapping weights.

Why this works: LoRA merging only changes the VALUES of existing Linear weights; the architecture (and therefore
the ONNX graph) is identical to the base model. The reference graph (onnx-community/granite-4.0-h-350m-ONNX,
ONNX Runtime GenAI-style ops) stores each HF Linear weight transposed as '<module>.MatMul.weight' and splits
the fused shared_mlp.input_linear into gate_proj / up_proj.

  # 1) merge adapter -> HF dir      python scripts/export_onnx.py merge --adapter runs/.../adapter --out models/merged/D
  # 2) HF dir -> ONNX fp32          python scripts/export_onnx.py onnx  --hf models/merged/D --out models/onnx/ft_fp32
  # 3) verify mapping on base       python scripts/export_onnx.py verify
  # 4) quantise                      python scripts/export_onnx.py quant --src models/onnx/ft_fp32 --out-root models/onnx --prefix ft
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "models/onnx_ref/onnx/model.onnx"
REF16 = ROOT / "models/onnx_ref/onnx/model_fp16.onnx"
BASE = ROOT / "models/granite-4.0-h-350m"


def hf_state(hf_dir):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32)
    return {k: v.detach().numpy() for k, v in m.state_dict().items()}


def onnx_value(name, sd, inter):
    """Return the numpy array the ONNX initializer `name` must hold, from HF state dict `sd`."""
    if name == "model.embed_tokens.weight":
        return sd["model.embed_tokens.weight"]
    if not name.endswith(".MatMul.weight"):
        return None
    mod = name[: -len(".MatMul.weight")]           # e.g. model.layers.3.mlp.gate_proj
    if ".mlp." in mod:
        layer, proj = mod.split(".mlp.")
        if proj in ("gate_proj", "up_proj"):
            W = sd[f"{layer}.shared_mlp.input_linear.weight"]      # (2*inter, hidden): [gate ; up]
            half = W[:inter] if proj == "gate_proj" else W[inter:]
            return half.T
        if proj == "down_proj":
            return sd[f"{layer}.shared_mlp.output_linear.weight"].T
    key = f"{mod}.weight".replace(".attn.", ".self_attn.")   # ONNX 'attn' == HF 'self_attn'
    if key in sd:
        return sd[key].T
    raise KeyError(f"no HF tensor for ONNX initializer {name}")


def build_onnx(hf_dir, out_dir, check_against_ref=False, ref=REF):
    from transformers import AutoConfig
    inter = AutoConfig.from_pretrained(hf_dir).intermediate_size
    sd = hf_state(hf_dir)
    m = onnx.load(str(ref), load_external_data=True)
    n_swapped, max_diff = 0, 0.0
    for init in m.graph.initializer:
        val = onnx_value(init.name, sd, inter)
        if val is None:
            continue
        ref = numpy_helper.to_array(init)
        assert ref.shape == val.shape, (init.name, ref.shape, val.shape)
        if check_against_ref:
            max_diff = max(max_diff, float(np.abs(ref.astype(np.float32) - val.astype(ref.dtype).astype(np.float32)).max()))
        init.CopyFrom(numpy_helper.from_array(val.astype(ref.dtype), init.name))
        n_swapped += 1
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("model.onnx*"):
        f.unlink()
    onnx.save_model(m, str(out_dir / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.onnx_data", size_threshold=1024)
    print(f"swapped {n_swapped} weight tensors -> {out_dir}" + (f"; max |ref - hf| = {max_diff:.3e}" if check_against_ref else ""))
    return n_swapped, max_diff


def merge(adapter, out):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    m = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    m.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
    print("merged ->", out)


def _strip_unused(model):
    used = {x for n in model.graph.node for x in n.input}
    for sg_holder in model.graph.node:                      # inputs used inside If/Loop subgraphs
        for a in sg_holder.attribute:
            if a.g is not None:
                used |= {x for n in a.g.node for x in n.input}
    keep = [i for i in model.graph.initializer if i.name in used]
    removed = len(model.graph.initializer) - len(keep)
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return removed


def _untie_lm_head(model):
    """lm_head = MatMul(h, Transpose(embed)) -> MatMul(h, constant embed.T) so embedding (Gather) and lm_head
    (MatMul) can each be quantised. Numerically identical in fp32."""
    emb = next(i for i in model.graph.initializer if i.name == "model.embed_tokens.weight")
    tr = next(n for n in model.graph.node if n.op_type == "Transpose" and n.input[0] == "model.embed_tokens.weight")
    mm = next(n for n in model.graph.node if n.op_type == "MatMul" and tr.output[0] in n.input)
    model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(numpy_helper.to_array(emb).T), "lm_head.weight_T"))
    mm.input[list(mm.input).index(tr.output[0])] = "lm_head.weight_T"
    model.graph.node.remove(tr)
    return model


def _int8_embedding(model):
    """Gather(fp32 embedding) -> Gather(int8 embedding) * Gather(per-row fp32 scale). Symmetric per-row int8."""
    from onnx import helper, TensorProto
    emb = next(i for i in model.graph.initializer if i.name == "model.embed_tokens.weight")
    W = numpy_helper.to_array(emb).astype(np.float32)
    scale = np.abs(W).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    Wq = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
    g = next(n for n in model.graph.node if n.op_type == "Gather" and n.input[0] == "model.embed_tokens.weight")
    ids, out = g.input[1], g.output[0]
    model.graph.initializer.remove(emb)
    model.graph.initializer.extend([numpy_helper.from_array(Wq, "embed_tokens.weight_int8"),
                                    numpy_helper.from_array(scale.astype(np.float32), "embed_tokens.scale")])
    idx = list(model.graph.node).index(g); model.graph.node.remove(g)
    new = [helper.make_node("Gather", ["embed_tokens.weight_int8", ids], ["/emb_q/g"], name="/emb_q/Gather", axis=0),
           helper.make_node("Cast", ["/emb_q/g"], ["/emb_q/f"], name="/emb_q/Cast", to=TensorProto.FLOAT),
           helper.make_node("Gather", ["embed_tokens.scale", ids], ["/emb_q/s"], name="/emb_q/GatherScale", axis=0),
           helper.make_node("Unsqueeze", ["/emb_q/s", "/emb_q/axes"], ["/emb_q/s2"], name="/emb_q/Unsq"),
           helper.make_node("Mul", ["/emb_q/f", "/emb_q/s2"], [out], name="/emb_q/Mul")]
    model.graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), "/emb_q/axes"))
    for k, n in enumerate(new):
        model.graph.node.insert(idx + k, n)
    return model


def _save(model, d):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("model.onnx*"):
        f.unlink()
    onnx.save_model(model, str(d / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.onnx_data", size_threshold=1024)


def quant(src, out_root, prefix):
    """FP16: all float weights -> float16 (IO + states fp16).
    INT8: dynamic quantisation, MatMul weights int8 per-channel (activations quantised at run time) + embedding
          stored int8 with per-row scale (Gather -> Cast -> Mul), since quantize_dynamic leaves Gather in fp32.
    INT4: weight-only MatMulNBits (block 32, symmetric) for MatMul + GatherBlockQuantized for the embedding.
    lm_head is untied from the embedding first so both get quantised; unused fp32 copies are stripped."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from onnxruntime.transformers.float16 import convert_float_to_float16
    src = Path(src) / "model.onnx"
    out_root = Path(out_root)
    # FP16: generic float16 conversion breaks the Mamba If-subgraphs, so FP16 is built separately by swapping
    # weights into the reference fp16 graph:  python scripts/export_onnx.py onnx16 --hf <hf dir> --out <dir>
    # untied fp32 intermediate
    tmp = out_root / f"_{prefix}_fp32_untied"
    _save(_untie_lm_head(onnx.load(str(src))), tmp)
    # INT8
    d8 = out_root / f"{prefix}_int8"; d8.mkdir(parents=True, exist_ok=True)
    tmp8 = out_root / f"_{prefix}_fp32_untied_emb8"
    _save(_int8_embedding(onnx.load(str(tmp / "model.onnx"))), tmp8)
    quantize_dynamic(str(tmp8 / "model.onnx"), str(d8 / "_raw.onnx"), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul"], use_external_data_format=True,
                     extra_options={"MatMulConstBOnly": True, "EnableSubgraph": True})
    m8 = onnx.load(str(d8 / "_raw.onnx")); r8 = _strip_unused(m8); shutil.rmtree(tmp8)
    for f in d8.glob("_raw*"):
        f.unlink()
    _save(m8, d8)
    # INT4
    q = MatMulNBitsQuantizer(onnx.load(str(tmp / "model.onnx")), block_size=32, is_symmetric=True, accuracy_level=4,
                             op_types_to_quantize=("MatMul", "Gather"), quant_axes=(("MatMul", 0), ("Gather", 1)))
    q.process()
    m4 = q.model.model; r4 = _strip_unused(m4)
    _save(m4, out_root / f"{prefix}_int4")
    shutil.rmtree(tmp)
    print(f"stripped unused initializers: int8 {r8}, int4 {r4}")
    for k in ("int8", "int4"):
        d = out_root / f"{prefix}_{k}"
        size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
        print(f"{k}: {d} {size/1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["merge", "onnx", "onnx16", "verify", "verify16", "quant"])
    ap.add_argument("--adapter"); ap.add_argument("--out"); ap.add_argument("--hf")
    ap.add_argument("--src"); ap.add_argument("--out-root", default=str(ROOT / "models/onnx")); ap.add_argument("--prefix", default="ft")
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a.adapter, a.out)
    elif a.cmd == "onnx":
        build_onnx(a.hf, a.out)
    elif a.cmd == "onnx16":
        build_onnx(a.hf, a.out, ref=REF16)
    elif a.cmd == "verify16":
        build_onnx(BASE, ROOT / "models/onnx/base_fp16", check_against_ref=True, ref=REF16)
    elif a.cmd == "verify":
        build_onnx(BASE, ROOT / "models/onnx/base_fp32", check_against_ref=True)
    elif a.cmd == "quant":
        quant(a.src, a.out_root, a.prefix)
