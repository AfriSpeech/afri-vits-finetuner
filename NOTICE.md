# Third-party components and licences

| Component | Use | Licence |
|---|---|---|
| [facebook/mms-tts-*](https://huggingface.co/facebook/mms-tts) (Meta MMS) | base models; the expanded bases are derivatives | **CC BY-NC 4.0** (non-commercial) |
| [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) | `vendor/finetune_hf_vits/` (patched) | MIT |
| [AfriSpeech/africa-g2p](https://github.com/AfriSpeech/africa-g2p) | IPA front-end and symbol inventory | see that repository |
| [Glottolog](https://glottolog.org) (CLDF release) | language classification used for `mapping/` | CC BY 4.0 |

**The expanded base models and any model fine-tuned from them inherit CC BY-NC 4.0 from MMS: non-commercial use only,
with attribution to Meta's MMS project.** The code in this repository (outside `vendor/`) is MIT.

Patches to the vendored trainer: `batch.get("speaker_id")` for single-speaker data, `buffer_rgba` instead of the removed
matplotlib `tostring_rgb`, LoRA injection (peft, non-grouped / non-dilated convolutions only), a separate discriminator
learning rate, and merged / adapter-only checkpoint export.
