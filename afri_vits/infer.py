"""Synthesise with a fine-tuned model."""
import json, os


def load(model_dir):
    import torch
    from transformers import VitsModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return VitsModel.from_pretrained(model_dir).eval().to(dev), AutoTokenizer.from_pretrained(model_dir)


def speak(model, tok, text, lang, speaker=None, speakers_json=None, seed=0, text_mode="ipa"):
    """speaker: an int id, or a label from speakers.json (multispeaker models). Returns (waveform ndarray, sampling_rate)."""
    import torch
    from .text import normalise
    sid = speaker
    if isinstance(speaker, str):
        sid = json.load(open(speakers_json))[speaker]
    ids = tok(normalise(text, lang, text_mode), return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        wav = model(**ids, speaker_id=sid).waveform if (sid is not None and model.config.num_speakers > 1) else model(**ids).waveform
    return wav[0].float().cpu().numpy(), model.config.sampling_rate
