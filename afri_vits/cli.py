"""afri-vits command line:  pick-base | prepare-data | finetune | speak | setup"""
import argparse, os, subprocess, sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="afri-vits")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pick-base", help="show which expanded MMS base is best for a language"); s.add_argument("lang")
    s = sub.add_parser("setup", help="build the monotonic-alignment extension (once)")
    s = sub.add_parser("prepare-data", help="dataset -> trainer parquet with africa-g2p IPA text")
    s.add_argument("--lang", required=True); s.add_argument("--dataset", required=True); s.add_argument("--out", required=True)
    s.add_argument("--config"); s.add_argument("--split", default="train"); s.add_argument("--audio-col", default="audio")
    s.add_argument("--text-col", default="text"); s.add_argument("--speaker-col"); s.add_argument("--duration-col")
    s.add_argument("--min-dur", type=float, default=1.0); s.add_argument("--max-dur", type=float, default=15.0)
    s.add_argument("--val", type=int, default=100); s.add_argument("--max-samples", type=int); s.add_argument("--keep-digits", action="store_true")
    s.add_argument("--text-mode", choices=["ipa", "clean"], default="ipa")
    s = sub.add_parser("finetune", help="fine-tune the best base for a language (LoRA or full)")
    s.add_argument("--lang", required=True); s.add_argument("--data", required=True, help="output dir of prepare-data")
    s.add_argument("--out", required=True); s.add_argument("--mode", choices=["lora", "full"], default="lora")
    s.add_argument("--base", help="override the automatic base (an MMS language code)"); s.add_argument("--steps", type=int, default=4000)
    s.add_argument("--batch", type=int, default=32); s.add_argument("--lr", type=float); s.add_argument("--lora-r", type=int, default=32)
    s.add_argument("--save-steps", type=int, default=1000); s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("speak", help="synthesise one sentence with a fine-tuned model")
    s.add_argument("--model", required=True); s.add_argument("--lang", required=True); s.add_argument("--text", required=True)
    s.add_argument("--out", default="out.wav"); s.add_argument("--speaker")
    a = p.parse_args(argv)

    if a.cmd == "pick-base":
        from .pick_base import describe; print(describe(a.lang))
    elif a.cmd == "setup":
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "finetune_hf_vits", "monotonic_align")
        os.makedirs(os.path.join(d, "monotonic_align"), exist_ok=True)
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=d, check=True)
    elif a.cmd == "prepare-data":
        from .data import prepare
        prepare(a.dataset, a.lang, a.out, config=a.config, split=a.split, audio_col=a.audio_col, text_col=a.text_col,
                speaker_col=a.speaker_col, duration_col=a.duration_col, min_dur=a.min_dur, max_dur=a.max_dur, val_n=a.val,
                drop_digits=not a.keep_digits, text_mode=a.text_mode, max_samples=a.max_samples)
    elif a.cmd == "finetune":
        from .train import finetune
        finetune(a.lang, a.data, a.out, mode=a.mode, base=a.base, steps=a.steps, batch=a.batch, lr=a.lr, lora_r=a.lora_r,
                 save_steps=a.save_steps, dry_run=a.dry_run)
    elif a.cmd == "speak":
        import soundfile as sf
        from .infer import load, speak
        m, t = load(a.model)
        spk = os.path.join(a.model, "speakers.json") if a.speaker else None
        w, sr = speak(m, t, a.text, a.lang, speaker=a.speaker, speakers_json=spk); sf.write(a.out, w, sr); print("wrote", a.out)


if __name__ == "__main__":
    main()
