"""Fine-tune an expanded base with LoRA or full fine-tuning (single- or multispeaker)."""
import os, sys, json, glob, subprocess
from .pick_base import pick_base, ensure_base

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor", "finetune_hf_vits")


def build_start_dir(base: str, base_dir: str, work_dir: str) -> str:
    """Training start point = expanded generator + the ORIGINAL MMS discriminator (fetched from facebook/mms-tts)."""
    if os.path.exists(os.path.join(work_dir, "model.safetensors")):
        return work_dir
    from huggingface_hub import hf_hub_download
    d_path = hf_hub_download("facebook/mms-tts", f"full_models/{base}/D_100000.pth")
    os.makedirs(work_dir, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(VENDOR, "convert_original_discriminator_checkpoint.py"),
                    "--checkpoint_path", d_path, "--generator_checkpoint_path", base_dir,
                    "--pytorch_dump_folder_path", work_dir], check=True, cwd=VENDOR, env=_env())
    return work_dir


def _env():
    e = dict(os.environ); e["PYTHONPATH"] = VENDOR + os.pathsep + e.get("PYTHONPATH", ""); return e


def make_config(*, start_dir, data_dir, out_dir, mode, steps, batch, lr, disc_lr, lora_r, lora_alpha, sample_text,
                multispeaker, max_tokens_length, save_steps, eval_steps, max_duration, workers):
    cfg = {
        "project_name": "afri_vits", "push_to_hub": False, "report_to": ["tensorboard"], "overwrite_output_dir": True,
        "output_dir": out_dir, "dataset_name": data_dir, "audio_column_name": "audio", "text_column_name": "text",
        "train_split_name": "train", "eval_split_name": "validation", "do_lower_case": False, "do_normalize": False,
        "full_generation_sample_text": sample_text, "max_duration_in_seconds": max_duration, "min_duration_in_seconds": 1.0,
        "max_tokens_length": max_tokens_length, "model_name_or_path": start_dir,
        "preprocessing_num_workers": workers, "dataloader_num_workers": 4,
        "do_train": True, "max_steps": steps, "num_train_epochs": 1000, "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": batch, "learning_rate": lr, "adam_beta1": 0.8, "adam_beta2": 0.99,
        "warmup_ratio": 0.02, "lr_decay": 0.9, "do_step_schedule_per_epoch": True,
        "do_eval": True, "eval_steps": eval_steps, "save_steps": save_steps, "per_device_eval_batch_size": 16, "max_eval_samples": 48,
        "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5, "weight_duration": 1, "weight_mel": 35,
        "fp16": True, "seed": 456,
    }
    if mode == "lora":
        cfg.update(use_lora=True, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, disc_learning_rate=disc_lr)
    if multispeaker:
        cfg.update(speaker_id_column_name="speaker_id", override_speaker_embeddings=True)
    return cfg


def finetune(lang, data_dir, out_dir, *, mode="lora", base=None, steps=4000, batch=32, lr=None, disc_lr=2e-5,
             lora_r=32, lora_alpha=64, sample_text=None, save_steps=1000, eval_steps=500, max_duration=15.0, workers=8,
             dry_run=False):
    assert mode in ("lora", "full"), mode
    info = pick_base(lang) if base is None else {"base": base, "name": lang, "quality": "user-chosen"}
    out_dir = os.path.abspath(out_dir); data_dir = os.path.abspath(data_dir)
    print(f"[afri-vits] {lang}: base mms-tts-{info['base']} ({info['quality']}); mode={mode}")
    base_dir = ensure_base(info["base"])
    start = build_start_dir(info["base"], base_dir, os.path.join(out_dir, "_start"))
    multispeaker = os.path.exists(os.path.join(data_dir, "speakers.json"))
    if sample_text is None:
        import pyarrow.parquet as pq
        sample_text = pq.read_table(os.path.join(data_dir, "validation-00000-of-00001.parquet"), columns=["text"]).column(0)[0].as_py()
    lr = lr if lr is not None else (2e-4 if mode == "lora" else 2e-5)
    cfg = make_config(start_dir=start, data_dir=data_dir, out_dir=out_dir, mode=mode, steps=steps, batch=batch, lr=lr, disc_lr=disc_lr,
                      lora_r=lora_r, lora_alpha=lora_alpha, sample_text=sample_text, multispeaker=multispeaker,
                      max_tokens_length=560, save_steps=save_steps, eval_steps=eval_steps, max_duration=max_duration, workers=workers)
    cfg_path = os.path.join(out_dir, "finetune_config.json")
    os.makedirs(out_dir, exist_ok=True); json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
    print(f"[afri-vits] config -> {cfg_path}" + ("  (multispeaker)" if multispeaker else ""))
    if dry_run:
        return cfg_path
    subprocess.run([sys.executable, os.path.join(VENDOR, "run_vits_finetuning.py"), cfg_path], check=True, cwd=VENDOR, env=_env())
    print(f"[afri-vits] done. Model: {out_dir} (load with transformers.VitsModel.from_pretrained)")
    return out_dir
