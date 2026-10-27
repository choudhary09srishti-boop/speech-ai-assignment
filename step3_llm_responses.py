"""STEP 3 - Feed transcripts to a small LLM and score its responses.

Task for the LLM: give a ONE-sentence Hindi summary of what the speaker said.
Gold response  = LLM output when given the REFERENCE (perfect) transcript.
Correctness    = word-overlap F1 between response(ASR transcript) and gold response.
Hallucination  = response contains a number that is not in the transcript, OR
                 more than THRESHOLD of its words do not appear in the transcript.
(These are automatic proxies; step 4 also creates a sheet for manual checking.)"""
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from utils import normalize_text, token_f1, sync_cuda

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.LLM_MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

SYSTEM = ("आप एक सहायक हैं। उपयोगकर्ता के पाठ का सार केवल हिंदी (देवनागरी लिपि) में एक छोटे वाक्य में लिखें। "
          "अंग्रेज़ी या रोमन लिपि का उपयोग न करें। केवल पाठ में दी गई जानकारी का उपयोग करें, नई जानकारी न जोड़ें।")

def respond(text: str):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    sync_cuda(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=config.LLM_MAX_NEW_TOKENS, do_sample=False)
    sync_cuda(); dt = time.perf_counter() - t0
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip(), dt

def has_digit(tokens):
    return {t for t in tokens if any(c.isdigit() for c in t)}

def hallucination(source: str, resp: str):
    src, rsp = normalize_text(source).split(), normalize_text(resp).split()
    if not rsp:
        return 0.0, True
    unsupported = sum(1 for w in rsp if w not in set(src)) / len(rsp)
    bad_numbers = len(has_digit(rsp) - has_digit(src)) > 0
    return unsupported, bool(bad_numbers or unsupported > config.HALLUCINATION_UNSUPPORTED_THRESHOLD)

respond("परीक्षण वाक्य")  # warm-up

# gold responses from the reference transcript
manifest = pd.read_csv(config.MANIFEST)
gold = {}
for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="gold"):
    gold[r.id], _ = respond(r.reference)

summary = []
for cond in config.CONDITIONS:
    asr_df = pd.read_csv(f"{config.RESULTS_DIR}/asr_{cond}.csv").fillna("")
    rows = []
    for _, r in tqdm(asr_df.iterrows(), total=len(asr_df), desc=cond):
        resp, llm_t = respond(r.hypothesis if r.hypothesis.strip() else " ")
        unsup, halluc = hallucination(r.hypothesis, resp)
        rows.append({"id": r.id, "condition": cond, "hypothesis": r.hypothesis,
                     "response": resp, "gold_response": gold[r.id],
                     "response_f1": token_f1(gold[r.id], resp),
                     "unsupported_ratio": unsup, "hallucination": halluc,
                     "asr_total_sec": r.total_sec, "llm_sec": llm_t,
                     "pipeline_sec": r.total_sec + llm_t})
    df = pd.DataFrame(rows)
    df.to_csv(f"{config.RESULTS_DIR}/llm_{cond}.csv", index=False, encoding="utf-8-sig")
    summary.append({"condition": cond, "mean_response_f1": df.response_f1.mean(),
                    "hallucination_rate": df.hallucination.mean(),
                    "mean_llm_sec": df.llm_sec.mean(), "mean_pipeline_sec": df.pipeline_sec.mean()})
    print(summary[-1])
pd.DataFrame(summary).to_csv(f"{config.RESULTS_DIR}/llm_summary.csv", index=False)
print("STEP 3 done.")
