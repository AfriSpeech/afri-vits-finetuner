"""Turn a user's speech dataset into the parquet layout the trainer expects (audio, IPA text, optional speaker_id)."""
import os, re, json, random, collections


def prepare(dataset, lang, out_dir, *, config=None, split="train", audio_col="audio", text_col="text",
            speaker_col=None, duration_col=None, min_dur=1.0, max_dur=15.0, val_n=100, drop_digits=True,
            text_mode="ipa", max_samples=None, seed=1234, num_proc=4):
    """dataset: a Hugging Face dataset id, a local folder of parquet files, or an 'audiofolder' (audio files + metadata.csv).

    Writes <out_dir>/{train,validation}-00000-of-00001.parquet and <out_dir>/prepare_report.json.
    With speaker_col set (and >1 distinct speakers) the files carry an int64 `speaker_id` column and <out_dir>/speakers.json
    maps the original labels to ids -> this is the multispeaker mode of the trainer.
    """
    import datasets
    from .text import normalise
    ds = datasets.load_dataset(dataset, config, split=split) if not os.path.isdir(dataset) else _load_local(dataset, split)
    for c in (audio_col, text_col) + ((speaker_col,) if speaker_col else ()):
        if c not in ds.column_names:
            raise SystemExit(f"column '{c}' not in dataset columns {ds.column_names}")
    ds = ds.cast_column(audio_col, datasets.Audio(decode=False))
    keep = [audio_col, text_col] + ([speaker_col] if speaker_col else []) + ([duration_col] if duration_col else [])
    ds = ds.select_columns(list(dict.fromkeys(keep)))
    if max_samples:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    def dur(ex):
        if duration_col: return float(ex[duration_col])
        import io, soundfile as sf
        b = ex[audio_col]["bytes"]
        if b is None: return 0.0
        i = sf.info(io.BytesIO(b)); return i.frames / i.samplerate

    def ok(ex):
        t = ex[text_col] or ""
        if drop_digits and re.search(r"\d", t): return False
        return len(t.strip()) >= 3 and min_dur <= dur(ex) <= max_dur

    n0 = len(ds); ds = ds.filter(ok, num_proc=num_proc)
    texts = [normalise(t, lang, text_mode) for t in ds[text_col]]
    good = [i for i, t in enumerate(texts) if len(t) >= 3]
    ds = ds.select(good); texts = [texts[i] for i in good]
    spk = None
    if speaker_col:
        labels = [str(x) for x in ds[speaker_col]]
        order = [s for s, _ in collections.Counter(labels).most_common()]
        if len(order) > 1:
            spk = {s: i for i, s in enumerate(order)}
            json.dump(spk, open(os.path.join(_mk(out_dir), "speakers.json"), "w"), ensure_ascii=False, indent=1)
            speaker_ids = [spk[s] for s in labels]
    feats = {"audio": datasets.Audio(sampling_rate=24000), "text": datasets.Value("string")}
    rows = {"audio": [ex[audio_col] for ex in ds], "text": texts}
    if spk:
        rows["speaker_id"] = speaker_ids; feats["speaker_id"] = datasets.Value("int64")
    out = datasets.Dataset.from_dict(rows, features=datasets.Features(feats))
    idx = list(range(len(out))); random.Random(seed).shuffle(idx)
    nval = min(val_n, max(1, len(out) // 20))
    val, tr = out.select(idx[:nval]), out.select(idx[nval:])
    _mk(out_dir)
    tr.to_parquet(os.path.join(out_dir, "train-00000-of-00001.parquet"))
    val.to_parquet(os.path.join(out_dir, "validation-00000-of-00001.parquet"))
    rep = {"language": lang, "text_mode": text_mode, "rows_in": n0, "rows_train": len(tr), "rows_validation": len(val),
           "speakers": len(spk) if spk else 1, "sample_text": texts[0][:80]}
    json.dump(rep, open(os.path.join(out_dir, "prepare_report.json"), "w"), indent=1)
    print(json.dumps(rep))
    return rep


def _mk(d):
    os.makedirs(d, exist_ok=True); return d


def _load_local(path, split):
    import glob, datasets
    pq = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if pq: return datasets.load_dataset("parquet", data_files=pq, split="train")
    return datasets.load_dataset("audiofolder", data_dir=path, split=split)
