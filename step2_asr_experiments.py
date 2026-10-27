"""STEP 2 - Run Whisper on 3 conditions: clean, noisy, noisy_denoised.
Measures WER, CER, latency, RTF. Saves results/asr_<condition>.csv"""
import time, sys
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import pipeline
import noisereduce as nr
import config
from utils import word_edits, char_edits, add_noise_snr, sync_cuda

manifest = pd.read_csv(config.MANIFEST)
device = 0 if torch.cuda.is_available() else -1
print("Device:", "GPU" if device == 0 else "CPU")
asr = pipeline("automatic-speech-recognition", model=config.WHISPER_MODEL, device=device)
kw = {"language": config.WHISPER_LANGUAGE, "task": "transcribe", **config.GEN_EXTRA}

# warm-up so the first timed clip is not unfairly slow
w, _ = sf.read(manifest.path[0], dtype="float32")
for _ in range(2):
    asr({"raw": w, "sampling_rate": config.SAMPLE_RATE}, generate_kwargs=kw)

summary = []
for cond in config.CONDITIONS:
    rows = []
    for i, r in tqdm(manifest.iterrows(), total=len(manifest), desc=cond):
        audio, sr = sf.read(r.path, dtype="float32")
        den_t = 0.0
        if cond in ("noisy", "noisy_denoised"):
            audio = add_noise_snr(audio, config.SNR_DB, seed=config.SEED + int(r.id))
        if cond == "noisy_denoised":
            t0 = time.perf_counter()
            audio = nr.reduce_noise(y=audio, sr=sr, stationary=True).astype(np.float32)
            den_t = time.perf_counter() - t0
        sync_cuda(); t0 = time.perf_counter()
        hyp = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs=kw)["text"]
        sync_cuda(); asr_t = time.perf_counter() - t0
        we, wn = word_edits(r.reference, hyp)
        ce, cn = char_edits(r.reference, hyp)
        rows.append({"id": r.id, "condition": cond, "reference": r.reference, "hypothesis": hyp,
                     "word_errors": we, "word_total": wn, "char_errors": ce, "char_total": cn,
                     "wer": we / max(wn, 1), "cer": ce / max(cn, 1),
                     "audio_sec": r.duration_sec, "denoise_sec": den_t, "asr_sec": asr_t,
                     "total_sec": den_t + asr_t, "rtf": (den_t + asr_t) / r.duration_sec})
    df = pd.DataFrame(rows)
    df.to_csv(f"{config.RESULTS_DIR}/asr_{cond}.csv", index=False, encoding="utf-8-sig")
    summary.append({"condition": cond,
                    "WER": df.word_errors.sum() / df.word_total.sum(),
                    "CER": df.char_errors.sum() / df.char_total.sum(),
                    "mean_total_latency_s": df.total_sec.mean(),
                    "p95_latency_s": df.total_sec.quantile(0.95),
                    "RTF": df.total_sec.sum() / df.audio_sec.sum()})
    print(summary[-1])

pd.DataFrame(summary).to_csv(f"{config.RESULTS_DIR}/asr_summary.csv", index=False)
import platform, transformers, json
info = {"whisper_model": config.WHISPER_MODEL, "snr_db": config.SNR_DB, "seed": config.SEED,
        "n_samples": len(manifest), "device": torch.cuda.get_device_name(0) if device == 0 else "CPU",
        "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version()}
json.dump(info, open(f"{config.RESULTS_DIR}/run_info.json", "w"), indent=2)
print("\nSTEP 2 done. Hardware:", torch.cuda.get_device_name(0) if device == 0 else "CPU")
