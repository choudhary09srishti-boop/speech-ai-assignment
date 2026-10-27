"""STEP 0 - Prove the whole chain works on ONE clip before doing 100."""
import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline
import config
from utils import normalize_text, word_edits, sync_cuda

print("GPU available:", torch.cuda.is_available())
device = 0 if torch.cuda.is_available() else -1

print("Loading ONE FLEURS Hindi test clip (first run downloads data, can take a few minutes)...")
ds = load_dataset(config.DATASET_NAME, config.LANG_CODE, split="test",
                  streaming=True, trust_remote_code=True)
ex = next(iter(ds))
audio = np.array(ex["audio"]["array"], dtype=np.float32)
sr = ex["audio"]["sampling_rate"]
print(f"Audio length: {len(audio)/sr:.2f}s, sample rate: {sr}")
print("Reference:", ex["raw_transcription"])

print(f"Loading {config.WHISPER_MODEL} ...")
asr = pipeline("automatic-speech-recognition", model=config.WHISPER_MODEL, device=device)
kw = {"language": config.WHISPER_LANGUAGE, "task": "transcribe", **config.GEN_EXTRA}
asr({"raw": audio, "sampling_rate": sr}, generate_kwargs=kw)   # warm-up (not timed)

sync_cuda(); t0 = time.perf_counter()
hyp = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs=kw)["text"]
sync_cuda(); dt = time.perf_counter() - t0

e, n = word_edits(ex["raw_transcription"], hyp)
print("Hypothesis:", hyp)
print(f"WER for this clip: {e/n:.2%}")
print(f"ASR time: {dt:.2f}s  |  RTF: {dt/(len(audio)/sr):.3f}")
print("\nSTEP 0 OK -> move on to step1_prepare_data.py")
