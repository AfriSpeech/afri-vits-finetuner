"""Enumerate every IPA character africa-g2p can emit, across all of its languages.

Sources (unioned):
  1. empirical: convert_to_ipa() over every grapheme, alphabet letter and example word in each language file
  2. the library's universal phoneme table (data/ipa_universal_graphemes.json), minus its internal ( ) _ markers
Writes data/african_ipa_symbols.json: {char: {"count": n_emitted, "languages": n_langs, "sources": [...]}}.
"""
import os, sys, json, glob, collections, unicodedata, time
import africa_g2p
from africa_g2p import convert_to_ipa, available_languages

pkg = os.path.dirname(africa_g2p.__file__)
codes = sorted(available_languages()) if callable(available_languages) else sorted(available_languages)
print("africa-g2p", getattr(africa_g2p, "__version__", "?"), "|", len(codes), "languages", flush=True)
cnt, langs, src = collections.Counter(), collections.defaultdict(set), collections.defaultdict(set)
fails, t0 = 0, time.time()
for li, code in enumerate(codes):
    f = os.path.join(pkg, "languages", f"{code}.json")
    if not os.path.exists(f): continue
    d = json.load(open(f))
    toks = set(d.get("graphemes", {})) | set(d.get("alphabet", []) or [])
    for e in d.get("examples", []) or []:
        toks.update((e if isinstance(e, str) else (e.get("text") or e.get("word") or "")).split())
    for tkn in toks:
        if not isinstance(tkn, str) or not tkn.strip(): continue
        try: out = convert_to_ipa(tkn, code)
        except Exception: fails += 1; continue
        for ch in out:
            if ch.isspace(): continue
            cnt[ch] += 1; langs[ch].add(code); src[ch].add("empirical")
    if li % 100 == 0: print(f"  {li}/{len(codes)} languages, {len(cnt)} symbols so far, {time.time()-t0:.0f}s", flush=True)
u = json.load(open(os.path.join(pkg, "data", "ipa_universal_graphemes.json")))["phonemes"]
for ph in u:
    for ch in ph:
        if ch in "()_ " or ch.isspace(): continue
        src[ch].add("universal_table"); cnt.setdefault(ch, 0)
res = {c: {"count": cnt[c], "languages": len(langs[c]), "sources": sorted(src[c])} for c in sorted(cnt, key=lambda c: (-len(langs[c]), c))}
os.makedirs("data", exist_ok=True)
json.dump(res, open("data/african_ipa_symbols.json", "w"), ensure_ascii=False, indent=1)
print(f"{len(res)} symbols | conversion failures {fails} | {time.time()-t0:.0f}s")
cat = collections.Counter(unicodedata.category(c) for c in res); print("unicode categories:", dict(cat))
print("emitted by >=20 languages:", sum(1 for v in res.values() if v["languages"] >= 20), "| by 1 language only:", sum(1 for v in res.values() if v["languages"] == 1),
      "| table-only (never emitted):", sum(1 for v in res.values() if v["sources"] == ["universal_table"]))
print("".join(res))
