#!/usr/bin/env python
"""Hero figure: in-distribution AUC vs out-of-distribution benign FPR
(@ uniform label-free conformal tau), 6 detector families x 2 models, bootstrap
95% CI error bars. AUDIT FIX M2: ALL points (incl. the proposed probe) and rho
are read from data/emnlp2026/wsf_tradeoff_uniform.json (the uniform-protocol
artifact); nothing is hardcoded. No GPU. Output: paper/fig_tradeoff.pdf
  python wsf_tradeoff_uniform.py && python make_tradeoff_figure.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
T = json.loads((REPO / "data/emnlp2026/wsf_tradeoff_uniform.json").read_text())
NAME = {"bert": "BERT", "fjd": "FJD", "gradsafe": "GradSafe",
        "hiddendetect_exact": "HiddenDetect", "jbshield": "JBShield-D",
        "ourprobe": "Our probe"}
MK = {"qwen_aligned": ("o", "#1f77b4", "Qwen2.5-7B"),
      "llama_aligned": ("s", "#d62728", "Llama-3.1-8B")}

fig, ax = plt.subplots(figsize=(5.4, 4.0))
for mdl, (mk, col, lab) in MK.items():
    first = True
    for meth in ["bert", "ourprobe", "hiddendetect_exact", "gradsafe",
                 "jbshield", "fjd"]:
        c = T["cells"][f"{meth}|{mdl}"]
        auc, fpr, (lo, hi) = c["auc"], c["ood_fpr"], c["ci"]
        ye = [[max(fpr - lo, 0)], [max(hi - fpr, 0)]]
        ax.errorbar(auc, fpr, yerr=ye, fmt=mk, color=col, ms=7, capsize=2,
                    elinewidth=0.8, mec="k", mew=0.4,
                    label=lab if first else None, zorder=3)
        first = False
        dx = 0.004 if meth != "ourprobe" else -0.004
        ha = "left" if meth != "ourprobe" else "right"
        ax.annotate(NAME[meth], (auc, fpr), (auc + dx, fpr + 0.025),
                    fontsize=7, color=col, ha=ha)

rho = T["spearman_rho_all_cells"]
p = T["spearman_p"]
ax.set_xlabel("In-distribution ROC-AUC", fontsize=10)
ax.set_ylabel("Out-of-distribution benign FPR\n(@ uniform conformal $\\tau$)",
              fontsize=10)
ax.set_xlim(0.38, 1.03)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.05, ls=":", c="gray", lw=0.8)
ax.text(0.40, 0.07, "target FPR 0.05", fontsize=6.5, color="gray")
ax.grid(alpha=0.25, lw=0.5)
ax.set_title("Higher in-distribution fit tends to worse benign over-refusal\n"
             f"(uniform protocol, 6 detector families; "
             f"$\\rho\\!=\\!{rho:.2f}$, $p\\!=\\!{p:.2g}$, descriptive)",
             fontsize=9.0)
ax.legend(loc="center left", fontsize=8, frameon=True)
fig.tight_layout()
out = REPO / "paper/fig_tradeoff.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"written: {out}  ({out.stat().st_size} bytes)  rho={rho} p={p:.2g}")
