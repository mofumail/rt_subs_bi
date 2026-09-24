#!/usr/bin/env python3
"""Expose decoder cross-attention in an ONNX Whisper export (for AlignAtt).

The onnx-community export of kotoba-whisper-bilingual keeps the cross-attention
softmax unfused but does not output it. This adds:

  input   xattn_heads       int64[k]           flat head ids (layer * n_heads + head)
  output  cross_attentions  float32[B,k,T,S]   gathered attention probabilities

to both branches of the merged decoder, so the policy reads only the heads it
uses (the full tensor for a 100-token prompt would be ~25 MB per step).

Usage:
  python scripts/patch_xattn.py --repo onnx-community/kotoba-whisper-bilingual-v1.0-ONNX \
      --out models/kotoba-whisper-bilingual-v1.0-xattn --dtypes q4f16 fp16 int8
"""

import argparse
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import onnx
from onnx import TensorProto, helper

DTYPE_SUFFIX = {
    "fp32": "",
    "fp16": "_fp16",
    "q4f16": "_q4f16",
    "q4": "_q4",
    "int8": "_int8",
    "uint8": "_uint8",
    "q8": "_quantized",
    "bnb4": "_bnb4",
}
SIDE_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
    "vocab.json",
    "merges.txt",
]


def fetch(repo: str, path: str, dest: Path, optional=False) -> bool:
    if dest.exists():
        return True
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"  fetch {path}", file=sys.stderr)
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
    except urllib.error.HTTPError as e:
        if optional and e.code == 404:
            return False
        raise
    tmp.rename(dest)
    return True


def layer_of(node_name: str) -> int:
    m = re.search(r"layers\.(\d+)/encoder_attn/", node_name)
    return int(m.group(1)) if m else -1


def patch_graph(g: onnx.GraphProto, prefix: str, n_heads_per_layer: int) -> str:
    """Adds Concat -> Gather -> Cast after the encoder_attn softmaxes of `g`
    and returns the name of the new (graph-local) output tensor."""
    sm = sorted(
        (n for n in g.node if n.op_type == "Softmax" and "encoder_attn" in n.name),
        key=lambda n: layer_of(n.name),
    )
    if not sm:
        raise RuntimeError(f"{prefix}: no encoder_attn Softmax nodes (attention fused?)")
    concat = f"{prefix}/xattn/concat"
    gathered = f"{prefix}/xattn/gathered"
    out = f"{prefix}/xattn/out"
    nodes = [
        helper.make_node("Concat", [n.output[0] for n in sm], [concat], axis=1, name=concat),
        helper.make_node("Gather", [concat, "xattn_heads"], [gathered], axis=1, name=gathered),
        helper.make_node("Cast", [gathered], [out], to=TensorProto.FLOAT, name=out),
    ]
    g.node.extend(nodes)
    g.output.append(
        helper.make_tensor_value_info(out, TensorProto.FLOAT, ["batch_size", "num_heads", "decoder_sequence_length", 1500])
    )
    return out


def patch(path_in: Path, path_out: Path, n_heads_per_layer: int):
    m = onnx.load(str(path_in), load_external_data=True)
    g = m.graph
    if any(o.name == "cross_attentions" for o in g.output):
        print(f"  {path_in.name}: already patched", file=sys.stderr)
        if path_in != path_out:
            shutil.copy(path_in, path_out)
        return
    g.input.append(helper.make_tensor_value_info("xattn_heads", TensorProto.INT64, ["num_heads"]))
    ifs = [n for n in g.node if n.op_type == "If"]
    if ifs:
        # Merged decoder: both branches (with / without past) get the output.
        (node,) = ifs
        for a in node.attribute:
            patch_graph(a.g, a.name, n_heads_per_layer)
        node.output.append("cross_attentions")
    else:
        local = patch_graph(g, "main", n_heads_per_layer)
        g.output.pop()
        g.node.append(helper.make_node("Identity", [local], ["cross_attentions"], name="xattn_identity"))
    g.output.append(
        helper.make_tensor_value_info(
            "cross_attentions", TensorProto.FLOAT, ["batch_size", "num_heads", "decoder_sequence_length", 1500]
        )
    )
    onnx.checker.check_model(m, full_check=False)
    size = m.ByteSize()
    if size > 1_900_000_000:
        onnx.save(m, str(path_out), save_as_external_data=True, location=path_out.name + "_data")
    else:
        onnx.save(m, str(path_out))
    print(f"  wrote {path_out} ({size / 1e6:.0f} MB)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="onnx-community/kotoba-whisper-bilingual-v1.0-ONNX")
    ap.add_argument("--out", default="models/kotoba-whisper-bilingual-v1.0-xattn")
    ap.add_argument("--dtypes", nargs="+", default=["q4f16"], choices=sorted(DTYPE_SUFFIX))
    ap.add_argument("--encoder-dtypes", nargs="*", default=None, help="defaults to --dtypes")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    cache = out / ".orig"
    for f in SIDE_FILES:
        fetch(args.repo, f, out / f, optional=True)
    cfg = json.loads((out / "config.json").read_text())
    n_heads = cfg["decoder_attention_heads"]

    for dt in args.encoder_dtypes if args.encoder_dtypes is not None else args.dtypes:
        name = f"encoder_model{DTYPE_SUFFIX[dt]}.onnx"
        fetch(args.repo, f"onnx/{name}", out / "onnx" / name)
        fetch(args.repo, f"onnx/{name}_data", out / "onnx" / f"{name}_data", optional=True)

    for dt in args.dtypes:
        name = f"decoder_model_merged{DTYPE_SUFFIX[dt]}.onnx"
        src = cache / name
        fetch(args.repo, f"onnx/{name}", src)
        fetch(args.repo, f"onnx/{name}_data", cache / f"{name}_data", optional=True)
        print(f"patch {name}", file=sys.stderr)
        patch(src, out / "onnx" / name, n_heads)

    # Record provenance so the runtime can refuse unpatched models early.
    (out / "rt_subs.json").write_text(
        json.dumps({"source": args.repo, "xattn": True, "decoder_heads": n_heads, "decoder_layers": cfg["decoder_layers"]}, indent=2)
    )


if __name__ == "__main__":
    main()
