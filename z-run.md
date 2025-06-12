Hydra is telling you there’s already a value at decoding.beam.beam_size, so just drop the leading +.

Pass 1 (corrected)

PYTORCH_ENABLE_MPS_FALLBACK=1 python speech_to_text_aed_chunked_infer.py \
  pretrained_name=nvidia/canary-1b-flash \
  audio_dir=lesson01_wav \
  audio_type=wav \
  chunk_len_in_secs=40.0 \
  batch_size=1 \
  compute_langs=true \
  decoding.beam.beam_size=4 \
  output_filename=_pass1_langid.json

Build the manifest for Pass 2

jq -c '
  if .lang=="en" then
    .taskname="s2t_translation" |
    .source_lang="en" |
    .target_lang="fr" |
    .save_src_text=true
  else
    .taskname="asr"
  end
' _pass1_langid.json > _pass2_manifest.json

Pass 2 (run translation + French ASR)

PYTORCH_ENABLE_MPS_FALLBACK=1 python speech_to_text_aed_chunked_infer.py \
  pretrained_name=nvidia/canary-1b-flash \
  dataset_manifest=_pass2_manifest.json \
  chunk_len_in_secs=40.0 \
  batch_size=1 \
  decoding.beam.beam_size=4 \
  output_filename=lesson01_fr_combined.json

lesson01_fr_combined.json will have:
	•	French dialogue from the original audio (taskname:"asr").
	•	French translations of the English narration (taskname:"s2t_translation", plus src_text holding the original English).