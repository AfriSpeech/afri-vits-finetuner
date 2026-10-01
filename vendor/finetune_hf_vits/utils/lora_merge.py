"""Merge LoRA into a plain, HF-loadable VitsModel WITHOUT copying the training model.

copy.deepcopy fails on modules that use the old-style torch.nn.utils.weight_norm (the speaker-conditioning layers that
`resize_speaker_embeddings` adds), so we build the merged state dict by hand and load it into a fresh VitsModel.
"""
import copy, re
import torch
from transformers import VitsModel


def merged_plain_model(raw):
    from peft.tuners.lora import LoraLayer
    sd = {n: p.detach().cpu().clone() for n, p in raw.state_dict().items()}
    for name, mod in raw.named_modules():
        if isinstance(mod, LoraLayer):
            adapter = next(iter(mod.lora_A.keys()))
            key = f"{name}.base_layer.weight"
            sd[key] = sd[key] + mod.get_delta_weight(adapter).detach().cpu().to(sd[key].dtype)
    out = {}
    for k, v in sd.items():
        if ".lora_" in k or k.startswith("discriminator."):
            continue
        k = k.replace(".base_layer.", ".")
        k = re.sub(r"\.weight_g$", ".parametrizations.weight.original0", k)   # old-style weight_norm -> parametrization
        k = re.sub(r"\.weight_v$", ".parametrizations.weight.original1", k)
        out[k] = v
    fresh = VitsModel(copy.deepcopy(raw.config))
    missing, unexpected = fresh.load_state_dict(out, strict=False)
    missing = [m for m in missing if not m.endswith(".attn.bias")]
    if missing or unexpected:
        raise RuntimeError(f"merged export mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return fresh.eval()
