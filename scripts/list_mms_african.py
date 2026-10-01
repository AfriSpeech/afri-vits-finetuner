"""List the African MMS-TTS checkpoints -> mapping/mms_african.json

African = an MMS-TTS language whose Glottolog entry is in macroarea Africa, plus any MMS language africa-g2p also supports.
(Using only the africa-g2p codes would drop checkpoints whose ISO code differs there, e.g. MMS `aka` vs africa-g2p `twi`/`fat`.)
"""
import os, io, csv, json, glob, urllib.request
import africa_g2p
from huggingface_hub import HfApi

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
cache = os.path.join(ROOT, "mapping", "_glottolog_cache"); os.makedirs(cache, exist_ok=True)
lp = os.path.join(cache, "languages.csv")
if not os.path.exists(lp):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv", lp)
area = {}
for r in csv.DictReader(open(lp, encoding="utf-8")):
    if r["ISO639P3code"] and (r["ISO639P3code"] not in area or r["Level"] == "language"):
        area[r["ISO639P3code"]] = r["Macroarea"]
pkg = os.path.dirname(africa_g2p.__file__)
g2p = {json.load(open(f))["code"] for f in glob.glob(f"{pkg}/languages/*.json")}
mms = {x.path.split("/")[-1] for x in HfApi().list_repo_tree("facebook/mms-tts", path_in_repo="full_models", recursive=False)}
EXCLUDE = {"krs": "facebook/mms-tts-krs has no weights file (broken upstream repo)"}
african = sorted(c for c in mms if (area.get(c) == "Africa" or c in g2p) and c not in EXCLUDE)
only_glot = sorted(c for c in african if c not in g2p)
json.dump({"n": len(african), "codes": african, "n_africa_g2p": len(g2p), "n_mms": len(mms),
           "added_by_glottolog_macroarea": only_glot, "excluded": EXCLUDE}, open(os.path.join(ROOT, "mapping", "mms_african.json"), "w"), indent=1)
print(len(african), "African MMS-TTS checkpoints |", len(only_glot), "found via Glottolog macroarea but not an africa-g2p code:", " ".join(only_glot))
