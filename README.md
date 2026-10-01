# afri-vits-finetuner

Fine-tune a text-to-speech model for **any African language supported by [africa-g2p](https://github.com/AfriSpeech/africa-g2p)** —
starting from the best-matching Meta MMS-TTS checkpoint, with IPA input.

MMS-TTS (VITS) covers ~1,100 languages, each as a separate model with its own character vocabulary. This repo makes
those models reusable for languages MMS does not cover:

1. **Expanded bases** — every African MMS-TTS checkpoint (362) has its vocabulary extended with all 190 IPA symbols
   africa-g2p can emit across its 750 languages, so any language's IPA text can be fed in.
2. **Language → base mapping** — each of the 750 africa-g2p languages is mapped to its closest MMS checkpoint
   (Glottolog tree distance, then IPA-inventory overlap, then geography). `afri-vits pick-base <lang>` shows the choice.
3. **Fine-tuning** — LoRA *or* full fine-tuning, single-speaker *or* multispeaker, on your own speech data.

> **Licence note.** MMS checkpoints are CC BY-NC 4.0. The expanded bases and every model fine-tuned from them are
> **non-commercial**. The code here is MIT. See [NOTICE.md](NOTICE.md).

## Quick start

```bash
pip install -r requirements.txt
python -m afri_vits.cli setup                      # build the monotonic-alignment extension (once)

python -m afri_vits.cli pick-base dag
# dag (Dagbani) -> facebook/mms-tts-maw expanded  [sister; tree distance 2, IPA overlap 0.933, 89 km away]

# 1. your data -> trainer format (IPA text via africa-g2p). HF dataset id, folder of parquet, or audiofolder + metadata.csv
python -m afri_vits.cli prepare-data --lang dag --dataset my-org/my-dagbani-speech --out data/dag
#    multispeaker: tell it which column holds the speaker label
python -m afri_vits.cli prepare-data --lang dag --dataset my-org/my-dagbani-speech --speaker-col speaker --out data/dag

# 2. fine-tune the automatically chosen base
python -m afri_vits.cli finetune --lang dag --data data/dag --out runs/dag --mode lora      # fast, ~4-7% of the weights
python -m afri_vits.cli finetune --lang dag --data data/dag --out runs/dag_full --mode full # all weights

# 3. speak
python -m afri_vits.cli speak --model runs/dag --lang dag --text "n tuma tuma ka n tooi" --out hello.wav
```

`finetune` picks the base, downloads the expanded generator from `AfriSpeech/afri-vits-bases` (or expands
`facebook/mms-tts-<code>` locally if offline), fetches the original MMS discriminator, and runs the trainer.
Override the base with `--base <mms code>` (see `mapping/language_to_base.json` for ranked alternatives).

## How the expansion works

VITS reads text through an embedding table with one row per symbol. For each MMS checkpoint we keep its original
vocabulary and ids, **append** the missing IPA symbols, and extend the table. The new rows start from the nearest existing
sound where there is one (IPA `j` ← the base's `y`, `ɡ` ← `g`, `ʃ` ← `s`; ~⅓ of them) or the mean embedding plus noise.
**This does not teach the model new sounds** — it makes the symbols readable. They are learned during fine-tuning, so expect
to need real data in your language. The tokenizer is switched to read IPA directly (`is_uroman=False`).

Only the generator is published; the discriminator (needed for GAN training) is fetched from `facebook/mms-tts` at
training time.

## Choosing LoRA or full fine-tuning

| | LoRA (`--mode lora`) | Full (`--mode full`) |
|---|---|---|
| Trains | low-rank adapters on text encoder, duration predictor, decoder + new token embeddings + (multispeaker) speaker layers | everything |
| lr (default) | 2e-4, discriminator 2e-5 | 2e-5 |
| Good when | little data (hours), you want to stay close to the base | more data, or the language is far from the base |
| Checkpoints | merged, directly loadable `VitsModel` every `--save-steps` | accelerate state each save; final model in `--out` |

Both default to 4,000 steps at batch 32. Short runs are enough: with ~10 h of audio, quality plateaus early, and longer
runs mostly add risk of overfitting.

## Multispeaker data

Give `prepare-data` a `--speaker-col`. Labels are mapped to ids (`speakers.json`), the base's speaker embedding is created
(it has none), and training conditions on speaker id. At inference pass `--speaker <label>`.

## Rebuilding the artefacts

```bash
python scripts/enumerate_ipa.py                    # every IPA char africa-g2p emits -> data/african_ipa_symbols.json
python scripts/make_symbol_list.py data/african_ipa_symbols.json symbols/african_ipa.json
python scripts/list_mms_african.py                 # African MMS-TTS checkpoints -> mapping/mms_african.json
python scripts/expand_base.py --all bases          # expand every base (generator only)
python scripts/build_mapping.py                    # -> mapping/language_to_base.{json,csv}
python scripts/publish_bases.py bases              # upload to AfriSpeech/afri-vits-bases
```

## Caveats

- Mapping quality varies. `mapping/language_to_base.csv` records, per language, `same_language`, `sister`, `close_relative`,
  `same_family` or `inventory_only`. Languages with no MMS relative (e.g. Zulu: no Nguni checkpoint exists) get the nearest
  Bantu base, which is a weak start.
- africa-g2p applies the target language's spelling rules to every word, so English loanwords come out as odd IPA
  (consistent, but not English-sounding).
- Rows containing digits are dropped by default (numbers are not spoken in the written form). Use `--keep-digits` to keep them.
- **Testing status.** LoRA fine-tuning was exercised on Dagbani (trained to step 3,000 of 4,000; pacing matched the real speaker, audio quality not formally evaluated). Full fine-tuning and multispeaker LoRA each hit a bug during testing (both fixed, see the commit log) and their re-runs were stopped before finishing, so those two paths are **unverified end to end**. Treat the whole repo as an early release.

## Credits

Meta MMS-TTS; [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) (vendored, patched);
[africa-g2p](https://github.com/AfriSpeech/africa-g2p); [Glottolog](https://glottolog.org).
