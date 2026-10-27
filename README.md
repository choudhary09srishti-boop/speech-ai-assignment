# Speech AI Assignment - Starter Project

Pipeline: Hindi audio -> Whisper Small (ASR) -> Qwen2.5-0.5B-Instruct (response) -> metrics.
Experiment: clean vs noisy (5 dB white noise) vs noisy + `noisereduce`.

## Run (Google Colab, GPU runtime)
1. Runtime > Change runtime type > T4 GPU
2. Upload this zip, unzip, `cd speech_project`
3. `pip install -r requirements.txt`
4. `python step0_feasibility.py`   (1 clip)
5. `python step1_prepare_data.py`  (100 clips)
6. `python step2_asr_experiments.py`
7. `python step3_llm_responses.py`
8. `python step4_report_tables.py`
Outputs are in `results/`. Or just open `run_on_colab.ipynb`.

## Definitions
- WER/CER: corpus-level edit distance / reference length, after normalising punctuation.
- RTF = processing time / audio duration (<1 is faster than real time). Denoising time is included.
- Response F1: word overlap with the response generated from the reference transcript.
- Hallucination (proxy): response has a number absent from the transcript OR >70% of its words are absent from it.
  Verify with `results/manual_review_sheet.csv`.

## Known limitations
- Whisper Tiny was tried first and failed on Hindi (Roman-script output, repetition loops); Small is used.
- Single language (Hindi), synthetic white noise, tiny LLM, automatic proxies for correctness/hallucination.
- WER is inflated by Hindi spelling variants (no spelling normalisation).
