"""STEP 1 - Download 100 FLEURS Hindi test clips, save as wav + manifest.csv."""
import os
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
import config

ds = load_dataset(config.DATASET_NAME, config.LANG_CODE, split="test", trust_remote_code=True)
print("Test split size:", len(ds))

rng = np.random.default_rng(config.SEED)
idx = rng.choice(len(ds), size=config.N_SAMPLES, replace=False)

rows = []
for k, i in enumerate(sorted(idx)):
    ex = ds[int(i)]
    audio = np.array(ex["audio"]["array"], dtype=np.float32)
    sr = ex["audio"]["sampling_rate"]
    assert sr == config.SAMPLE_RATE, f"unexpected sample rate {sr}"
    path = os.path.join(config.CLEAN_DIR, f"{k:03d}.wav")
    sf.write(path, audio, sr)
    rows.append({"id": f"{k:03d}", "path": path,
                 "reference": ex["raw_transcription"],
                 "duration_sec": round(len(audio) / sr, 3)})

df = pd.DataFrame(rows)
df.to_csv(config.MANIFEST, index=False)
print(df.head())
print(f"Saved {len(df)} clips. Total audio: {df.duration_sec.sum()/60:.1f} min")
