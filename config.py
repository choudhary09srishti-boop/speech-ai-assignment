"""Central settings. Change things HERE, not inside the step scripts."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")

# ---- Dataset ----
DATASET_NAME = "google/fleurs"
LANG_CODE = "hi_in"          # Hindi (India) in FLEURS
N_SAMPLES = 100              # assignment minimum
SEED = 42
SAMPLE_RATE = 16000          # Whisper expects 16 kHz

# ---- ASR ----
WHISPER_MODEL = "openai/whisper-small"  # tiny gave Roman-script garbage + loops for Hindi
WHISPER_LANGUAGE = "hindi"

# ---- Noise experiment ----
SNR_DB = 5                   # lower = noisier. 5 dB is clearly noisy.
CONDITIONS = ["clean", "noisy", "noisy_denoised"]

# ---- Response generator (small LLM) ----
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_MAX_NEW_TOKENS = 160
HALLUCINATION_UNSUPPORTED_THRESHOLD = 0.7   # see README for the definition

# ---- Manual review ----
MANUAL_REVIEW_N = 30

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Generation guard against repetition loops ----
GEN_EXTRA = {"max_new_tokens": 256, "no_repeat_ngram_size": 5}
