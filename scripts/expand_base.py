"""Expand one MMS-TTS checkpoint with the canonical pan-African IPA symbol list (generator only).

    python scripts/expand_base.py <mms_code> <out_root>        # e.g. ewe bases
    python scripts/expand_base.py --all <out_root> [--workers N]   # every African MMS-TTS language in mapping/mms_african.json

Writes <out_root>/<code>/ : a plain transformers VitsModel + tokenizer whose vocab = the original MMS vocab (ids unchanged)
+ the missing canonical symbols appended; new embedding rows start at the nearest existing sound or mean+noise; the
tokenizer is set to read IPA directly (is_uroman=False). The discriminator is NOT included: training fetches it from
facebook/mms-tts (see afri_vits/train.py).
"""
import os, sys, json, shutil, unicodedata, collections, traceback, time
import torch
from huggingface_hub import snapshot_download
from transformers import VitsModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = json.load(open(os.path.join(HERE, "..", "symbols", "african_ipa.json")))["symbols"]
NEAR = {"j": "y", "ɡ": "g", "ɲ": "n", "ɳ": "n", "ɴ": "n", "ɱ": "m", "ʃ": "s", "ʒ": "z", "ʂ": "s",
        "ʐ": "z", "ɕ": "s", "ʑ": "z", "ç": "s", "θ": "t", "ð": "d", "β": "b", "ɓ": "b", "ɗ": "d",
        "ɠ": "g", "ƙ": "k", "ɟ": "y", "c": "k", "q": "k", "ɧ": "h", "ɦ": "h", "ʔ": "h", "ʕ": "h",
        "ɾ": "r", "ɽ": "r", "ɹ": "r", "ɻ": "r", "ʁ": "r", "ʀ": "r", "ɬ": "l", "ɮ": "l", "ɫ": "l",
        "ʎ": "l", "ɭ": "l", "ʍ": "w", "ɥ": "y", "χ": "x", "ɢ": "g"}


def expand(code, out_root, seed=0):
    out = os.path.join(out_root, code)
    if os.path.exists(os.path.join(out, "ipa_expansion.json")):
        return "skip"
    src = snapshot_download(f"facebook/mms-tts-{code}")
    vocab = json.load(open(f"{src}/vocab.json")); n0 = len(vocab)
    model = VitsModel.from_pretrained(src)
    emb = model.text_encoder.embed_tokens; w = emb.weight.data.clone()
    assert w.shape[0] == n0 and max(vocab.values()) == n0 - 1, (code, w.shape[0], n0, max(vocab.values()))
    add = [c for c in SYMBOLS if c not in vocab]
    for c in add: vocab[c] = len(vocab)
    V = len(vocab)
    g = torch.Generator().manual_seed(seed + sum(map(ord, code)))
    std, mean = w.std(), w.mean(0)
    rows, how = [], collections.Counter()
    for c in add:
        parent = NEAR.get(c)
        if parent is None:
            nfd = unicodedata.normalize("NFD", c)[0]
            parent = nfd if nfd != c else None
        if parent and parent in vocab and vocab[parent] < n0 and parent != c:
            rows.append(w[vocab[parent]] + torch.randn(w.shape[1], generator=g) * std * 0.1); how["nearest_sound"] += 1
        else:
            rows.append(mean + torch.randn(w.shape[1], generator=g) * std * 0.5); how["mean_noise"] += 1
    if rows:
        model.text_encoder.embed_tokens = torch.nn.Embedding(V, w.shape[1], padding_idx=emb.padding_idx)
        model.text_encoder.embed_tokens.weight.data = torch.cat([w, torch.stack(rows)], 0)
    model.config.vocab_size = V
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    model.save_pretrained(out)
    for f in ("special_tokens_map.json", "tokenizer_config.json", "added_tokens.json", "preprocessor_config.json"):
        if os.path.exists(f"{src}/{f}"): shutil.copy(f"{src}/{f}", f"{out}/{f}")
    json.dump(vocab, open(f"{out}/vocab.json", "w"), ensure_ascii=False, indent=1)
    tc = json.load(open(f"{out}/tokenizer_config.json")); was_uroman = tc.get("is_uroman", False)
    tc["is_uroman"] = False
    dec = tc.get("added_tokens_decoder", {})
    if str(n0) in dec: dec[str(V)] = dec.pop(str(n0))          # <unk> moves to the end
    json.dump(tc, open(f"{out}/tokenizer_config.json", "w"), ensure_ascii=False, indent=1)
    if os.path.exists(f"{out}/added_tokens.json"): json.dump({"<unk>": V}, open(f"{out}/added_tokens.json", "w"))
    # smoke test: loads, tokenizes IPA, forward pass is finite
    m2 = VitsModel.from_pretrained(out).eval(); tok = AutoTokenizer.from_pretrained(out)
    ids = tok("ka ɲa t͡ʃa ɔː", return_tensors="pt")
    with torch.no_grad(): wav = m2(**ids).waveform
    ok = bool(torch.isfinite(wav).all())
    json.dump({"mms_code": code, "base": f"facebook/mms-tts-{code}", "orig_vocab_size": n0, "vocab_size": V, "n_added": len(add),
               "added_init": dict(how), "orig_is_uroman": was_uroman, "smoke_forward_finite": ok,
               "added": add}, open(f"{out}/ipa_expansion.json", "w"), ensure_ascii=False, indent=1)
    return f"{code}: {n0}->{V} (+{len(add)}) uroman={was_uroman} ok={ok}"


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--all":
        out_root = args[1]; codes = json.load(open(os.path.join(HERE, "..", "mapping", "mms_african.json")))["codes"]
        part, nparts = (int(args[args.index("--part") + 1]), int(args[args.index("--part") + 2])) if "--part" in args else (0, 1)
        codes = codes[part::nparts]; t0 = time.time(); fails = []
        for i, c in enumerate(codes):
            try: print(f"[{i+1}/{len(codes)} {time.time()-t0:.0f}s]", expand(c, out_root), flush=True)
            except Exception as e:
                fails.append(c); print(f"[{i+1}/{len(codes)}] FAIL {c}: {repr(e)[:160]}", flush=True)
        print("done; failures:", fails)
    else:
        print(expand(args[0], args[1]))
