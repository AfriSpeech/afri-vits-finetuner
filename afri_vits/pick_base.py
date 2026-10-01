"""Pick (and fetch) the best expanded MMS base for an africa-g2p language code."""
import json, os, sys
from . import BASES_REPO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPING = os.path.join(ROOT, "mapping", "language_to_base.json")


def load_mapping():
    return json.load(open(MAPPING, encoding="utf-8"))


def pick_base(lang: str) -> dict:
    m = load_mapping()
    if lang not in m:
        raise SystemExit(f"'{lang}' is not an africa-g2p language code. Close matches: "
                         + ", ".join(k for k in m if k.startswith(lang[:2]))[:200])
    return {"language": lang, **m[lang]}


def ensure_base(base: str, cache_dir: str = None) -> str:
    """Return a local directory holding the expanded generator for MMS language `base`.
    Tries the Hugging Face repo first; if that is unavailable, expands facebook/mms-tts-<base> locally."""
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "afri-vits", "bases")
    local = os.path.join(cache_dir, base)
    if os.path.exists(os.path.join(local, "ipa_expansion.json")):
        return local
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(BASES_REPO, allow_patterns=[f"{base}/*"], local_dir=cache_dir)
        if os.path.exists(os.path.join(local, "ipa_expansion.json")):
            return local
    except Exception as e:
        print(f"[afri-vits] could not fetch {BASES_REPO}/{base} ({type(e).__name__}); expanding facebook/mms-tts-{base} locally", file=sys.stderr)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from expand_base import expand
    expand(base, cache_dir)
    return local


def describe(lang: str) -> str:
    i = pick_base(lang)
    jac = lambda x: "n/a" if x is None else x
    alts = ", ".join(f"{a['base']} (d={a['tree_distance']}, jaccard {jac(a['ipa_jaccard'])})" for a in i["alternatives"])
    km = f", {i['km']} km away" if i.get("km") is not None else ""
    return (f"{lang} ({i['name']}) -> facebook/mms-tts-{i['base']} expanded  [{i['quality']}; tree distance {i['tree_distance']}, "
            f"IPA overlap {jac(i['ipa_jaccard'])}{km}]\n  alternatives: {alts}")
