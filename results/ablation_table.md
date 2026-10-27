| condition | WER | CER | mean_total_latency_s | p95_latency_s | RTF | mean_response_f1 | hallucination_rate | mean_llm_sec | mean_pipeline_sec |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (clean audio) | 0.533 | 0.250 | 2.747 | 4.409 | 0.256 | 0.202 | 0.310 | 4.095 | 6.842 |
| Noisy (5 dB SNR), no denoising | 0.746 | 0.431 | 2.807 | 4.481 | 0.262 | 0.146 | 0.360 | 4.032 | 6.839 |
| Noisy + noisereduce (improvement) | 0.791 | 0.497 | 2.883 | 4.810 | 0.269 | 0.134 | 0.320 | 3.898 | 6.781 |