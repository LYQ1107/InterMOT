#!/usr/bin/env python
"""N11 final analysis: paired stats, same-ID TTE survival, figures."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(".")
OUT = ROOT / "outputs/n11"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEQS = [
    "dancetrack0074",
    "dancetrack0075",
    "dancetrack0080",
    "dancetrack0082",
    "dancetrack0083",
    "dancetrack0086",
    "dancetrack0087",
    "dancetrack0096",
    "dancetrack0098",
    "dancetrack0099",
]
METRICS = ["HOTA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]


def load_per_seq(path: Path) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["method"], {})[r["sequence"]] = {
            m: float(r[m]) for m in METRICS
        }
    return out


n10 = load_per_seq(ROOT / "outputs/n10/eval/train/per_sequence_metrics.csv")
n11 = load_per_seq(OUT / "eval/train/per_sequence_metrics.csv")
n11x = load_per_seq(OUT / "eval/train_extra/per_sequence_metrics.csv")


def paired_stats(a: dict, b: dict) -> list:
    rows = []
    for m in METRICS:
        d = np.array([a[s][m] - b[s][m] for s in SEQS])
        rng = np.random.default_rng(20260809)
        boots = [
            np.mean(rng.choice(d, size=len(d), replace=True))
            for _ in range(10000)
        ]
        ci = np.percentile(boots, [2.5, 97.5])
        if np.all(d == 0):
            p = 1.0
            dz = 0.0
        else:
            try:
                p = float(stats.wilcoxon(d).pvalue)
            except ValueError:
                p = float("nan")
            dz = float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
        imp = int(np.sum(d > 0))
        deg = int(np.sum(d < 0))
        same = int(np.sum(d == 0))
        rows.append(
            {
                "metric": m,
                "mean_delta": round(float(np.mean(d)), 5),
                "median_delta": round(float(np.median(d)), 5),
                "ci_low": round(float(ci[0]), 5),
                "ci_high": round(float(ci[1]), 5),
                "wilcoxon_p": round(p, 6),
                "cohens_dz": round(dz, 4),
                "improved": imp,
                "degraded": deg,
                "unchanged": same,
            }
        )
    return rows


comparisons = [
    ("N11 Local B8 vs AUTO-P b0", "local_native0_decay_b8_post", n11, "pairwise_b0_post", n10),
    ("N11 Local B8 vs N10 Global B8", "local_native0_decay_b8_post", n11, "human_b8_post", n10),
    ("N10 Global B8 vs AUTO-P b0", "human_b8_post", n10, "pairwise_b0_post", n10),
    ("N11 Local-perm B8 vs AUTO-P b0", "local_perm_b8_post", n11x, "pairwise_b0_post", n10),
    ("N11 Local-Evidence B8 vs AUTO-P b0", "local_native0_evidence_b8_post", n11x, "pairwise_b0_post", n10),
]
stat_rows = []
for name, ma, sa, mb, sb in comparisons:
    for r in paired_stats(sa[ma], sb[mb]):
        stat_rows.append({"comparison": name, **r})
with (OUT / "statistical_results.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
    w.writeheader()
    w.writerows(stat_rows)

# Same-ID TTE survival from collateral analysis
col = list(csv.DictReader(open(OUT / "collateral_analysis.csv")))
variants = ["global_b8", "local_perm_b8", "local_decay_b8", "local_decay_b4", "local_decay_b2", "local_decay_b1", "auto_b8"]
horizons = [1, 3, 5, 10, 15, 20, 25, 30]
surv_rows = []
for v in variants:
    vals = np.array([float(r["same_tte"]) for r in col if r["variant"] == v])
    surv_rows.append(
        {
            "variant": v,
            "n": int(len(vals)),
            **{f"t{h}": round(float(np.mean(vals >= h)) * 100.0, 2) for h in horizons},
        }
    )
with (OUT / "same_id_tte_survival.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(surv_rows[0].keys()))
    w.writeheader()
    w.writerows(surv_rows)

# Figures
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})

# 1. persistence curve
horizons_p = [1, 3, 5, 10, 30]
curves = {
    "N8 (reported)": (horizons_p[:3], [5.0, 5.0, 5.0]),
    "N9 B8": (horizons_p[:3], [53.75, 40.0, 37.5]),
    "N10 Global B8": (horizons_p, [51.25, 37.5, 30.0, 28.57, 19.48]),
    "N11 Local-decay B8": (horizons_p, [47.5, 26.25, 18.75, 14.29, 10.39]),
    "AUTO branch B8": (horizons_p, [9.52, 9.52, 9.52, 7.5, 4.88]),
}
fig, ax = plt.subplots(figsize=(6, 4))
for name, (xs, ys) in curves.items():
    ax.plot(xs, ys, marker="o", label=name)
ax.set_xlabel("frames after interaction")
ax.set_ylabel("same-ID retention (%)")
ax.set_title("Corrected-ID persistence curve (calibration)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "persistence_curve.png")
plt.close(fig)

# 2. same-ID TTE survival
fig, ax = plt.subplots(figsize=(6, 4))
for row in surv_rows:
    ax.plot(horizons, [row[f"t{h}"] for h in horizons], marker="o", label=row["variant"])
ax.set_xlabel("frames after interaction")
ax.set_ylabel("P(same-ID TTE >= t) (%)")
ax.set_title("Same-ID time-to-next-error survival (calibration)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "same_id_tte_survival.png")
plt.close(fig)

# 3. collateral damage (30-frame aggregate)
agg = {
    "global_b8": (594, 1727),
    "local_perm_b8": (594, 1727),
    "local_decay_b8": (315, 1275),
    "local_decay_b4": (130, 651),
    "local_decay_b2": (78, 311),
    "local_decay_b1": (51, 142),
    "auto_b8": (0, 0),
}
names = list(agg)
selfs = [agg[n][0] for n in names]
others = [agg[n][1] for n in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - 0.2, selfs, width=0.4, label="self (target) assignment changes")
ax.bar(x + 0.2, others, width=0.4, label="unrelated assignment changes")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("total assignment changes / 80 events / 30 frames")
ax.set_title("Collateral assignment-change aggregate (30-frame window)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIG / "collateral_damage_curve.png")
plt.close(fig)

# 4. target gain vs collateral scatter
fig, ax = plt.subplots(figsize=(6, 4))
for v, c, lab in [
    ("global_b8", "#d62728", "N10 Global B8"),
    ("local_perm_b8", "#ff9896", "N11 Local-perm B8"),
    ("local_decay_b8", "#1f77b4", "N11 Local-decay B8"),
    ("auto_b8", "#2ca02c", "AUTO branch B8"),
]:
    xs = [float(r["target_err_30"]) for r in col if r["variant"] == v]
    ys = [float(r["unrelated_err_30"]) for r in col if r["variant"] == v]
    ax.scatter(xs, ys, s=10, alpha=0.55, color=c, label=lab)
ax.set_xlabel("target future errors / interaction (30 frames)")
ax.set_ylabel("unrelated future errors / interaction (30 frames)")
ax.set_title("Target gain vs collateral damage per interaction")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "target_vs_collateral.png")
plt.close(fig)

# 5. quality-cost curve
global_metrics = {
    "B1": (41.375, 34.286, 45.730, 794),
    "B2": (41.376, 34.282, 45.730, 798),
    "B4": (41.220, 34.020, 45.629, 809),
    "B8": (41.157, 33.913, 45.537, 819),
}
local_metrics = {
    "B1": (41.342, 34.233, 45.640, 792),
    "B2": (41.342, 34.229, 45.640, 797),
    "B4": (41.344, 34.226, 45.641, 807),
    "B8": (41.297, 34.143, 45.573, 823),
}
bs = ["B1", "B2", "B4", "B8"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for name, m in [("N10 Global", global_metrics), ("N11 Local-decay", local_metrics)]:
    axes[0].plot(bs, [m[b][0] for b in bs], marker="o", label=f"{name} HOTA")
    axes[0].plot(bs, [m[b][1] for b in bs], marker="s", ls="--", label=f"{name} AssA")
    axes[0].plot(bs, [m[b][2] for b in bs], marker="^", ls=":", label=f"{name} IDF1")
axes[0].axhline(41.402, color="k", lw=0.8, label="AUTO-P HOTA")
axes[0].axhline(34.330, color="gray", lw=0.8, ls="--", label="AUTO-P AssA")
axes[0].set_xlabel("budget (corrections / 100 frames)")
axes[0].set_ylabel("metric")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)
for name, m in [("N10 Global", global_metrics), ("N11 Local-decay", local_metrics)]:
    axes[1].plot(bs, [m[b][3] for b in bs], marker="o", label=f"{name} IDSW")
axes[1].axhline(772, color="k", lw=0.8, label="AUTO-P IDSW")
axes[1].set_xlabel("budget (corrections / 100 frames)")
axes[1].set_ylabel("IDSW")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.suptitle("Quality-cost curve over budget (calibration)")
fig.tight_layout()
fig.savefig(FIG / "quality_cost_curve.png")
plt.close(fig)

(OUT / "n11_analysis_summary.json").write_text(
    json.dumps(
        {
            "statistical_results": "outputs/n11/statistical_results.csv",
            "same_id_tte_survival": "outputs/n11/same_id_tte_survival.csv",
            "figures": [p.name for p in sorted(FIG.glob("*.png"))],
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
print("analysis OK")
