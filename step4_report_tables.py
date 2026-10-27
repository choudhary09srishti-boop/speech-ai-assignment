"""STEP 4 - Build the ablation table, a plot, and the manual-review sheet."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

a = pd.read_csv(f"{config.RESULTS_DIR}/asr_summary.csv")
l = pd.read_csv(f"{config.RESULTS_DIR}/llm_summary.csv")
t = a.merge(l, on="condition")
names = {"clean": "Baseline (clean audio)", "noisy": f"Noisy ({config.SNR_DB} dB SNR), no denoising",
         "noisy_denoised": "Noisy + noisereduce (improvement)"}
t["condition"] = t.condition.map(names)
t.to_csv(f"{config.RESULTS_DIR}/ablation_table.csv", index=False)

cols = list(t.columns)
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, r in t.iterrows():
    md.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
open(f"{config.RESULTS_DIR}/ablation_table.md", "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t.plot.bar(x="condition", y=["WER", "CER"], ax=ax[0], rot=15); ax[0].set_title("Error rates")
t.plot.bar(x="condition", y=["mean_total_latency_s", "mean_llm_sec"], ax=ax[1], rot=15)
ax[1].set_title("Latency (s)")
plt.tight_layout(); plt.savefig(f"{config.RESULTS_DIR}/comparison.png", dpi=150)

# manual review sheet (clean condition): you fill the last 2 columns by hand
d = pd.read_csv(f"{config.RESULTS_DIR}/llm_clean.csv")
s = d.sample(n=config.MANUAL_REVIEW_N, random_state=config.SEED)[
    ["id", "hypothesis", "response", "gold_response", "hallucination"]].rename(
    columns={"hallucination": "auto_hallucination_flag"})
s["manual_correct_0_or_1"] = ""
s["manual_hallucination_0_or_1"] = ""
s.to_csv(f"{config.RESULTS_DIR}/manual_review_sheet.csv", index=False, encoding="utf-8-sig")
print("\nSaved ablation_table.*, comparison.png, manual_review_sheet.csv in results/")
