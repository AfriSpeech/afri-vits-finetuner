"""Upload the expanded bases to one Hugging Face repo, one folder per MMS language code.

    python scripts/publish_bases.py <bases_dir> [--repo AfriSpeech/afri-vits-bases] [--public]
Uses huggingface_hub's resumable large-folder uploader (safe to re-run; skips what is already uploaded).
"""
import os, sys, json
from huggingface_hub import HfApi

bases = sys.argv[1]; repo = "AfriSpeech/afri-vits-bases"
if "--repo" in sys.argv: repo = sys.argv[sys.argv.index("--repo") + 1]
api = HfApi()
api.create_repo(repo, repo_type="model", private="--public" not in sys.argv, exist_ok=True)
codes = sorted(d for d in os.listdir(bases) if os.path.exists(os.path.join(bases, d, "ipa_expansion.json")))
print(len(codes), "bases ready ->", repo)
api.upload_large_folder(folder_path=bases, repo_id=repo, repo_type="model", ignore_patterns=["*.log", "*.tmp"], num_workers=8)
print("uploaded")
