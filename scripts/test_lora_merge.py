"""Numerical check of the LoRA merge (single- and multispeaker): merged plain model == LoRA'd training model.
    cd vendor/finetune_hf_vits && PYTHONPATH=. python ../../scripts/test_lora_merge.py <training start dir> [n_speakers]
"""
import sys, torch
import torch.distributed.tensor  # peft/torch quirk
from peft import LoraConfig, inject_adapter_in_model
from utils import VitsModelForPreTraining
from utils.lora_merge import merged_plain_model

start, nspk = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
m = VitsModelForPreTraining.from_pretrained(start); del m.discriminator
if nspk > 1: m.resize_speaker_embeddings(nspk, 256)
targets = [n for n, mod in m.named_modules() if isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)) and getattr(mod, "groups", 1) == 1
           and tuple(getattr(mod, "dilation", (1,))) == (1,) and n.split(".")[0] in ("text_encoder", "duration_predictor", "decoder")
           and "cond" not in n and "embed" not in n and not n.endswith("conv_post")]
m = inject_adapter_in_model(LoraConfig(r=32, lora_alpha=64, target_modules=targets), m)
torch.manual_seed(0)
for n, p in m.named_parameters():
    if "lora_B" in n: torch.nn.init.normal_(p, std=0.02)          # make the LoRA delta non-zero
m.eval()
fresh = merged_plain_model(m)
ids = torch.randint(1, 40, (2, 30)); mask = torch.ones_like(ids)
pm = mask.unsqueeze(-1).float()
with torch.no_grad():
    a = m.text_encoder(input_ids=ids, padding_mask=pm, attention_mask=mask).last_hidden_state
    b = fresh.text_encoder(input_ids=ids, padding_mask=pm, attention_mask=mask).last_hidden_state
    w = fresh(input_ids=ids[:1], speaker_id=0 if nspk > 1 else None).waveform
print(f"speakers={nspk} | LoRA layers {len(targets)} | text-encoder max |LoRA - merged| = {(a-b).abs().max():.2e} | waveform finite: {bool(torch.isfinite(w).all())}")
assert (a - b).abs().max() < 1e-4
print("MERGE OK")
