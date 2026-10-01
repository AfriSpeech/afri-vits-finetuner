"""Synthesise comparison samples from a fine-tuned checkpoint.

    python scripts/synth_samples.py <model_dir> <prepared_data_dir> <out_dir> [n=6] [--speaker-ids 0,1]
Writes <out_dir>/synth/NN.wav (model), <out_dir>/truth/NN.wav (the real recording of the same held-out sentence),
sentences.json. Sentences come from the held-out validation split, so the model did not train on them.
"""
import sys, os, io, json, torch, numpy as np, soundfile as sf, pyarrow.parquet as pq
from transformers import VitsModel, AutoTokenizer

model_dir, data_dir, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 6
spk_ids = [None]
if "--speaker-ids" in sys.argv:
    spk_ids = [int(x) for x in sys.argv[sys.argv.index("--speaker-ids") + 1].split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = VitsModel.from_pretrained(model_dir).eval().to(dev); tok = AutoTokenizer.from_pretrained(model_dir)
rows = pq.read_table(f"{data_dir}/validation-00000-of-00001.parquet").to_pylist()
rows = [r for r in rows if 30 <= len(r["text"]) <= 110][:n]          # moderate-length sentences
os.makedirs(f"{out}/truth", exist_ok=True)
info = []
for i, r in enumerate(rows):
    open(f"{out}/truth/{i:02d}.wav", "wb").write(r["audio"]["bytes"])
    true_dur = sf.info(io.BytesIO(r["audio"]["bytes"])).duration
    for sid in spk_ids:
        d = f"{out}/synth" + ("" if sid is None else f"_speaker{sid}"); os.makedirs(d, exist_ok=True)
        ids = tok(r["text"], return_tensors="pt").to(dev)
        torch.manual_seed(1234 + i)
        with torch.no_grad():
            w = (m(**ids, speaker_id=sid) if sid is not None else m(**ids)).waveform[0].float().cpu().numpy()
        sf.write(f"{d}/{i:02d}.wav", w, m.config.sampling_rate)
        info.append({"i": i, "speaker": sid, "text": r["text"], "synth_s": round(len(w) / m.config.sampling_rate, 2), "true_s": round(true_dur, 2),
                     "rms": round(float(np.sqrt((w ** 2).mean())), 3), "finite": bool(np.isfinite(w).all())})
json.dump(info, open(f"{out}/sentences.json", "w"), ensure_ascii=False, indent=1)
for x in info: print(x["i"], x["speaker"], f"synth {x['synth_s']}s vs real {x['true_s']}s | rms {x['rms']} finite {x['finite']} |", x["text"][:60])
