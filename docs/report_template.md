# Speech AI Assignment Report

## Part A - Architecture Research (Task 1)
1. Overview of the two models
2. Comparison of the seven topics (table)
3. Comparative diagram
4. Limitations (latency, Indic, noise)
5. Proposed improvement + how to evaluate it

## Part B - Implementation (Task 2)
1. Pipeline: audio -> Whisper-tiny -> Qwen2.5-0.5B -> response (diagram)
2. Setup: dataset (FLEURS Hindi test, 100 clips, seed 42), hardware (copy from step 2 output), versions
3. Metrics definitions: WER, CER, RTF, response F1, hallucination (automatic proxy + manual check of 30)
4. Results: paste results/ablation_table.md and comparison.png
5. Analysis: did denoising help? Latency cost of denoising? Examples of failures (pick 3-5 rows from CSVs)
6. Limitations: single language, synthetic white noise, tiny models, proxy metrics, no speech output
7. How to reproduce (README steps)
