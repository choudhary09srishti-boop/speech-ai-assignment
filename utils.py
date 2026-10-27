"""Helper functions: text normalisation, WER/CER, noise, timing."""
import unicodedata
import numpy as np


def normalize_text(text: str) -> str:
    """Lower-case, remove punctuation/symbols (incl. the Hindi danda), collapse spaces.
    We drop characters by Unicode category so Devanagari vowel signs are kept."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        out.append(" " if cat.startswith(("P", "S")) else ch)
    return " ".join("".join(out).split())


def edit_distance(ref_seq, hyp_seq) -> int:
    """Levenshtein distance between two sequences (lists of words or characters)."""
    n, m = len(ref_seq), len(hyp_seq)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_seq[i - 1] == hyp_seq[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def word_edits(ref: str, hyp: str):
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    return edit_distance(r, h), len(r)


def char_edits(ref: str, hyp: str):
    r, h = list(normalize_text(ref).replace(" ", "")), list(normalize_text(hyp).replace(" ", ""))
    return edit_distance(r, h), len(r)


def add_noise_snr(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise so that signal-to-noise ratio equals snr_db."""
    rng = np.random.default_rng(seed)
    audio = audio.astype(np.float32)
    sig_power = float(np.mean(audio ** 2)) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def token_f1(ref: str, hyp: str) -> float:
    """Word-overlap F1 between two texts (0..1)."""
    from collections import Counter
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    if not r or not h:
        return 0.0
    common = sum((Counter(r) & Counter(h)).values())
    if common == 0:
        return 0.0
    p, rec = common / len(h), common / len(r)
    return 2 * p * rec / (p + rec)


def sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
