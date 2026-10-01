"""Map every africa-g2p language to the African MMS-TTS checkpoint that is the best starting point for fine-tuning it.

Closeness, in order:
  1. tree distance in Glottolog's classification (edges between the two languages through their lowest common ancestor)
  2. overlap (Jaccard) of the IPA symbols africa-g2p produces for the two languages
  3. geographic distance between the languages
Languages Glottolog cannot place (or with no relative among the MMS checkpoints) fall back to 2 then 3 (quality 'inventory_only').

    python scripts/build_mapping.py            # -> mapping/language_to_base.json (+ .csv)
"""
import os, re, io, csv, json, glob, math, urllib.request, collections

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
CACHE = os.path.join(ROOT, "mapping", "_glottolog_cache"); os.makedirs(CACHE, exist_ok=True)
BASE = "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/"
for f in ("languages.csv", "classification.nex"):
    if not os.path.exists(os.path.join(CACHE, f)):
        urllib.request.urlretrieve(BASE + f, os.path.join(CACHE, f))

# ---- Glottolog: languoids + classification trees -------------------------------------------------
langs = {r["ID"]: r for r in csv.DictReader(open(os.path.join(CACHE, "languages.csv"), encoding="utf-8"))}
iso2glot = {}
for gid, r in langs.items():
    code = r["ISO639P3code"]
    if not code: continue
    target = gid if r["Level"] != "dialect" else (r["Language_ID"] or gid)
    if code not in iso2glot or r["Level"] == "language": iso2glot[code] = target

parent = {}
def parse_newick(s):
    stack, i, cur = [], 0, None
    tok = re.compile(r"\s*([(),;])|\s*([A-Za-z0-9_'-]*)(?::[0-9.eE+-]+)?")
    children = []
    stack = [[]]
    pos = 0
    while pos < len(s):
        c = s[pos]
        if c == "(": stack.append([]); pos += 1
        elif c == ",": pos += 1
        elif c == ")":
            kids = stack.pop(); pos += 1
            m = re.match(r"([A-Za-z0-9_]*)(?::[0-9.eE+-]+)?", s[pos:]); label = m.group(1); pos += m.end()
            for k in kids: parent[k] = label
            stack[-1].append(label)
        elif c == ";": pos += 1
        else:
            m = re.match(r"([A-Za-z0-9_]+)(?::[0-9.eE+-]+)?", s[pos:])
            if not m: pos += 1; continue
            stack[-1].append(m.group(1)); pos += m.end()
    return stack[0]

for name, nw in re.findall(r"tree\s+(\S+)\s*=\s*(?:\[&[UR]\])?\s*(\(.*?;)", open(os.path.join(CACHE, "classification.nex")).read(), flags=re.S):
    parse_newick(nw)

def path_of(g):
    p = [g]
    while p[-1] in parent: p.append(parent[p[-1]])
    return p[::-1]                       # root -> ... -> g

def tree_distance(a, b):
    pa, pb = path_of(a), path_of(b)
    if pa[0] != pb[0]: return None       # different top-level families
    k = 0
    while k < min(len(pa), len(pb)) and pa[k] == pb[k]: k += 1
    return (len(pa) - k) + (len(pb) - k)

def km(a, b):
    try:
        la1, lo1, la2, lo2 = (float(langs[a]["Latitude"]), float(langs[a]["Longitude"]), float(langs[b]["Latitude"]), float(langs[b]["Longitude"]))
    except (KeyError, ValueError): return None
    p = math.pi / 180
    h = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))

# ---- africa-g2p: per-language IPA inventory ----------------------------------------------------
import africa_g2p
from africa_g2p import convert_to_ipa
pkg = os.path.dirname(africa_g2p.__file__)
meta, inv = {}, {}
for f in glob.glob(f"{pkg}/languages/*.json"):
    d = json.load(open(f)); code = d["code"]; meta[code] = d
    toks = set(d.get("graphemes", {})) | set(d.get("alphabet", []) or [])
    s = set()
    for t in toks:
        if isinstance(t, str) and t.strip():
            try: s.update(ch for ch in convert_to_ipa(t, code) if not ch.isspace())
            except Exception: pass
    inv[code] = s
mms = json.load(open(os.path.join(ROOT, "mapping", "mms_african.json")))["codes"]
def inv_of(M):
    """IPA symbols for an MMS base. Bases that are not africa-g2p codes (e.g. MMS `aka`) borrow the union of the africa-g2p
    languages they cover (Akan -> Twi + Fante)."""
    if M in inv: return inv[M]
    gM = iso2glot.get(M); s = set()
    if gM:
        for L2 in inv:
            gL2 = iso2glot.get(L2)
            if gL2 and (gL2 == gM or gM in path_of(gL2)): s |= inv[L2]
    return s
base_inv = {M: inv_of(M) for M in mms}
jac = lambda a, M: len(inv[a] & base_inv[M]) / max(1, len(inv[a] | base_inv[M]))

def quality(d, same):
    if same: return "same_language"
    if d is None: return "inventory_only"
    return "sister" if d <= 2 else "close_relative" if d <= 6 else "same_family"

out = {}
for L in sorted(meta):
    gL = iso2glot.get(L)
    cands = []
    for M in mms:
        gM = iso2glot.get(M)
        same = (M == L) or (gL is not None and gL == gM)
        d = 0 if same else (tree_distance(gL, gM) if gL and gM else None)
        cands.append((same, d, jac(L, M), km(gL, gM) if gL and gM else None, M))
    related = [c for c in cands if c[1] is not None]
    pool = sorted(related, key=lambda c: (c[1], -c[2], c[3] if c[3] is not None else 1e9)) if related else \
           sorted(cands, key=lambda c: (-c[2], c[3] if c[3] is not None else 1e9))
    best = pool[0]
    out[L] = {"name": meta[L].get("name"), "family_g2p": meta[L].get("family"), "glottocode": gL,
              "base": best[4], "quality": quality(best[1], best[0]), "tree_distance": best[1],
              "ipa_jaccard": round(best[2], 3) if base_inv[best[4]] else None, "km": None if best[3] is None else round(best[3]),
              "alternatives": [{"base": c[4], "tree_distance": c[1], "ipa_jaccard": round(c[2], 3) if base_inv[c[4]] else None} for c in pool[1:4]]}
json.dump(out, open(os.path.join(ROOT, "mapping", "language_to_base.json"), "w"), ensure_ascii=False, indent=1)
with open(os.path.join(ROOT, "mapping", "language_to_base.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["language", "name", "base", "quality", "tree_distance", "ipa_jaccard", "km"])
    for L, v in out.items(): w.writerow([L, v["name"], v["base"], v["quality"], v["tree_distance"], v["ipa_jaccard"], v["km"]])
print(len(out), "languages mapped to", len({v["base"] for v in out.values()}), "distinct bases | quality:", dict(collections.Counter(v["quality"] for v in out.values())))
print("bases with no inventory (jaccard unavailable):", sum(1 for v in base_inv.values() if not v), "| glottolog-placed:", sum(1 for v in out.values() if v["glottocode"]), "of", len(out))
for L in ("twi", "fat", "aka", "ewe", "dag", "gur", "kus", "hau", "yor", "swh", "zul", "wol", "bam", "kik", "lin", "amh"):
    if L in out: v = out[L]; print(f"  {L} {v['name'][:14]:14s} -> {v['base']} ({v['quality']}, dist {v['tree_distance']}, jaccard {v['ipa_jaccard']}, {v['km']} km)")
