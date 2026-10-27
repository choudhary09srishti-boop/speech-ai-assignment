"""STEP 6 (GPU, ~6-8 min) - Research extension: reduce the LLM latency bottleneck.
Measured bottleneck: the LLM stage takes ~4 s of ~6.8 s per clip (baseline limit: 160 new tokens).
Improvement tested: cap generation length (48 and 96 new tokens), clean audio only.
Compares latency, response F1 (vs the SAME gold responses as the baseline) and hallucination."""
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from utils import normalize_text, token_f1, sync_cuda

LIMITS = [48, 96]
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.LLM_MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

SYSTEM = ("आप एक सहायक हैं। उपयोगकर्ता के पाठ का सार केवल हिंदी (देवनागरी लिपि) में एक छोटे वाक्य में लिखें। "
          "अंग्रेज़ी या रोमन लिपि का उपयोग न करें। केवल पाठ में दी गई जानकारी का उपयोग करें, नई जानकारी न जोड़ें।")

def respond(text, max_new):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    sync_cuda(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    sync_cuda(); dt = time.perf_counter() - t0
    gen = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip(), dt, len(gen)

def digits(tokens):
    return {t for t in tokens if any(c.isdigit() for c in t)}

def hallucination(source, resp):
    src, rsp = normalize_text(source).split(), normalize_text(resp).split()
    if not rsp:
        return True
    unsupported = sum(1 for w in rsp if w not in set(src)) / len(rsp)
    return bool(len(digits(rsp) - digits(src)) > 0 or unsupported > config.HALLUCINATION_UNSUPPORTED_THRESHOLD)

base = pd.read_csv(f"{config.RESULTS_DIR}/llm_clean.csv").fillna("")
respond("परीक्षण वाक्य", 16)  # warm-up

base_len = [len(tok(r).input_ids) for r in base.response]
rows = [{"setting": f"baseline ({config.LLM_MAX_NEW_TOKENS} tokens)", "mean_llm_sec": base.llm_sec.mean(),
         "mean_pipeline_sec": base.pipeline_sec.mean(), "mean_response_f1": base.response_f1.mean(),
         "hallucination_rate": base.hallucination.mean(),
         "hit_token_limit_rate": float(sum(n >= config.LLM_MAX_NEW_TOKENS - 2 for n in base_len) / len(base_len))}]

for lim in LIMITS:
    res = []
    for _, r in tqdm(base.iterrows(), total=len(base), desc=f"limit {lim}"):
        resp, dt, n = respond(r.hypothesis if r.hypothesis.strip() else " ", lim)
        res.append({"id": r.id, "response": resp, "gold_response": r.gold_response,
                    "response_f1": token_f1(r.gold_response, resp),
                    "hallucination": hallucination(r.hypothesis, resp), "llm_sec": dt,
                    "pipeline_sec": r.asr_total_sec + dt, "hit_limit": n >= lim})
    d = pd.DataFrame(res)
    d.to_csv(f"{config.RESULTS_DIR}/llm_clean_maxtok{lim}.csv", index=False, encoding="utf-8-sig")
    rows.append({"setting": f"max {lim} tokens", "mean_llm_sec": d.llm_sec.mean(),
                 "mean_pipeline_sec": d.pipeline_sec.mean(), "mean_response_f1": d.response_f1.mean(),
                 "hallucination_rate": d.hallucination.mean(), "hit_token_limit_rate": d.hit_limit.mean()})

t = pd.DataFrame(rows)
t.to_csv(f"{config.RESULTS_DIR}/latency_improvement.csv", index=False)
cols = list(t.columns)
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, r in t.iterrows():
    md.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
open(f"{config.RESULTS_DIR}/latency_improvement.md", "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
