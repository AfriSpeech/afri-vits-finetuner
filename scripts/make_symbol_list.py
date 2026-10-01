"""Curate the raw africa-g2p emission list (data/african_ipa_symbols.json) into the canonical expansion list.

Keeps IPA letters, modifier letters, combining marks and glottal/ejective apostrophes + stress marks; drops punctuation, math
signs and non-IPA scripts that leaked in from non-Latin language entries. a-z always comes first.
    python scripts/make_symbol_list.py data/african_ipa_symbols.json symbols/african_ipa.json
"""
import json, sys, string, unicodedata

def keep(c):
    cat = unicodedata.category(c); name = unicodedata.name(c, "")
    if cat[0] in "PSZC": return c in "ʼˈˌ’'"
    if any(s in name for s in ("HEBREW", "HANGUL", "CYRILLIC", "ARABIC", "CJK")): return False
    return cat in ("Ll", "Lm", "Lo", "Mn")

raw = json.load(open(sys.argv[1]))
curated = [c for c in raw if keep(c)]
symbols = list(string.ascii_lowercase) + [c for c in curated if c not in string.ascii_lowercase]
dropped = [c for c in raw if not keep(c)]
json.dump({"symbols": symbols, "dropped_non_ipa": dropped, "n_languages": 750,
           "note": "every character africa-g2p convert_to_ipa emits over all languages + its universal phoneme table, curated"},
          open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print(len(symbols), "symbols | dropped:", "".join(dropped))
