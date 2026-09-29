#!/usr/bin/env python3
"""Write the N72R17 frozen Goal state, decision, and human-readable report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


GOAL = "Human Identity Representation Probe"
QUESTION_EN = "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?"
QUESTION_ZH = "用户点一下这个人以后，我们到底能不能认住他？"
PRIMARY = "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE"
ENCODER_OSNET = "osnet_x1_0_market1501"
ENCODER_CLIP = "openai_clip_vit_b32_zero_shot"


def h100(summary: dict[str, object]) -> dict[str, object]:
    return summary["horizons"]["H100"]


def value(summary: dict[str, object], key: str) -> float | None:
    raw = h100(summary).get(key)
    return float(raw) if raw is not None else None


def pct(raw: object) -> str:
    return "NA" if raw is None else f"{100.0 * float(raw):.2f}%"


def dec(raw: object) -> str:
    return "NA" if raw is None else f"{float(raw):.6f}"


def fmt_ci(row: dict[str, object]) -> str:
    ci = row.get("hard_negative_win_rate_sequence_cluster_ci95", {})
    return f"[{pct(ci.get('lower'))}, {pct(ci.get('upper'))}]"


def single_decision(aggregate: dict[str, object]) -> dict[str, object]:
    val = aggregate["encoder_benchmark"]
    osnet = val[ENCODER_OSNET]["val"]
    clip = val[ENCODER_CLIP]["val"]
    osnet_h100 = value(osnet, "hard_negative_win_rate")
    clip_h100 = value(clip, "hard_negative_win_rate")
    clip_delta = clip_h100 - osnet_h100 if clip_h100 is not None and osnet_h100 is not None else None

    memory = aggregate["memory_baselines"]
    candidates: list[tuple[str, dict[str, object], float | None]] = []
    for encoder in (ENCODER_OSNET, ENCODER_CLIP):
        single = value(val[encoder]["val"], "hard_negative_win_rate")
        for variant, summary in memory[encoder]["val"]["variants"].items():
            current = value(summary, "hard_negative_win_rate")
            delta = current - single if current is not None and single is not None else None
            candidates.append((f"{encoder}:{variant}", summary, delta))
    best_memory = max(candidates, key=lambda item: item[2] if item[2] is not None else float("-inf"))
    best_name, best_summary, best_delta = best_memory
    best_h100 = value(best_summary, "hard_negative_win_rate")
    best_ci = h100(best_summary).get("hard_negative_win_rate_sequence_cluster_ci95", {})
    clip_ci = h100(clip).get("hard_negative_win_rate_sequence_cluster_ci95", {})
    max_single = max(item for item in (osnet_h100, clip_h100) if item is not None)

    # These gates are declared before reading the result and are deliberately
    # identity-specific, not MOT/HOTA thresholds.
    encoder_limited = bool(
        clip_h100 is not None
        and clip_delta is not None
        and clip_h100 >= 0.50
        and clip_delta >= 0.10
        and clip_ci.get("lower") is not None
        and float(clip_ci["lower"]) > 0.40
    )
    memory_promising = bool(
        best_delta is not None
        and best_delta >= 0.05
        and best_h100 is not None
        and best_h100 >= 0.50
        and best_ci.get("lower") is not None
        and float(best_ci["lower"]) > 0.40
    )
    human_conditioned_required = bool(max_single >= 0.50 and not encoder_limited and not memory_promising)
    if encoder_limited:
        decision = "ENCODER_LIMITED"
        rationale = "The public CLIP encoder makes a material H100 hard-negative improvement over OSNet under the frozen protocol; a better identity backbone is the first limiting factor."
    elif memory_promising:
        decision = "TEMPORAL_MEMORY_PROMISING"
        rationale = "A causal frozen memory diagnostic materially improves H100 identity wins and retains a positive sequence-cluster lower bound, warranting a learned human-conditioned memory stage."
    elif human_conditioned_required:
        decision = "HUMAN_CONDITIONED_IDENTITY_REQUIRED"
        rationale = "At least one frozen encoder carries measurable identity signal, but simple causal memory baselines do not meet the material-improvement gate; the next scientific question is a learned human-conditioned identity memory."
    else:
        decision = "IDENTITY_SIGNAL_TOO_WEAK"
        rationale = "Both frozen encoder families and simple diagnostic memories remain below the predeclared identity-signal gate against hard competitors."
    next_memory_authorized = decision in {"TEMPORAL_MEMORY_PROMISING", "HUMAN_CONDITIONED_IDENTITY_REQUIRED"}
    return {
        "decision": decision,
        "rationale": rationale,
        "osnet_h100": osnet_h100,
        "clip_h100": clip_h100,
        "clip_delta_over_osnet": clip_delta,
        "best_memory_variant": best_name,
        "best_memory_h100": best_h100,
        "best_memory_delta_over_same_encoder": best_delta,
        "next_human_conditioned_identity_stage_authorized": next_memory_authorized,
        "next_association_stage_authorized": False,
        "gates": {
            "encoder_limited": "CLIP H100 >= 0.50, CLIP-OSNet delta >= 0.10, and CLIP sequence-CI lower > 0.40",
            "memory_promising": "best memory delta >= 0.05, best H100 >= 0.50, and best sequence-CI lower > 0.40",
            "human_conditioned_required": "best single-anchor H100 >= 0.50 while the first two gates fail",
            "identity_signal_too_weak": "otherwise",
        },
    }


def metric_table(aggregate: dict[str, object], section: str) -> str:
    rows = [
        "| Encoder/variant | Split | H20 win | H50 win | H100 win | H100 CI95 | Median margin | MRR | Rank-1 | Rank-2 | Rank-3 |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    source = aggregate[section]
    for encoder in (ENCODER_OSNET, ENCODER_CLIP):
        for split in ("train", "val"):
            if section == "encoder_benchmark":
                variants = [("Single Anchor", source[encoder][split])]
            else:
                variants = [(variant, summary) for variant, summary in source[encoder][split]["variants"].items()]
            for variant, summary in variants:
                h20 = summary["horizons"]["H20"]
                h50 = summary["horizons"]["H50"]
                h100_row = summary["horizons"]["H100"]
                rows.append(
                    f"| `{encoder}` / {variant} | {split} | {pct(h20.get('hard_negative_win_rate'))} | {pct(h50.get('hard_negative_win_rate'))} | {pct(h100_row.get('hard_negative_win_rate'))} | {fmt_ci(h100_row)} | {dec(h100_row.get('median_margin'))} | {dec(h100_row.get('mrr'))} | {pct(h100_row.get('rank_1_accuracy'))} | {pct(h100_row.get('rank_2_accuracy'))} | {pct(h100_row.get('rank_3_accuracy'))} |"
                )
    return "\n".join(rows)


def secondary_table(summary: dict[str, object]) -> str:
    rows = [
        "| Time gap | samples | win rate | mean margin | median margin | P10 | P25 | P75 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("1_5", "6_20", "21_50", "51_100"):
        row = summary["time_gap_bins"][label]
        rows.append(
            f"| {label.replace('_', '–')} | {row['samples']} | {pct(row.get('hard_negative_win_rate'))} | {dec(row.get('mean_margin'))} | {dec(row.get('median_margin'))} | {dec(row.get('p10_margin'))} | {dec(row.get('p25_margin'))} | {dec(row.get('p75_margin'))} |"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    decision = single_decision(aggregate)
    now = datetime.now(timezone.utc).isoformat()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    final_goal = {
        "stage": "N72R17",
        "goal": GOAL,
        "central_question": QUESTION_EN,
        "central_question_zh": QUESTION_ZH,
        "primary_endpoint": PRIMARY,
        "horizons": [20, 50, 100],
        "sam3_required": False,
        "mot_evaluation_required": False,
        "training_required": False,
        "next_association_stage_authorized": False,
        "goal_frozen": True,
        "inherited_from": "N72R16",
    }
    (output_dir / "FINAL_GOAL.json").write_text(json.dumps(final_goal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_result = {
        "stage": "N72R17",
        "goal": GOAL,
        "central_question": QUESTION_EN,
        "primary_endpoint": PRIMARY,
        "decision": decision["decision"],
        "rationale": decision["rationale"],
        "encoder_comparison": {
            "osnet_val_h100": decision["osnet_h100"],
            "clip_val_h100": decision["clip_h100"],
            "clip_delta_over_osnet": decision["clip_delta_over_osnet"],
        },
        "best_memory": {
            "variant": decision["best_memory_variant"],
            "val_h100": decision["best_memory_h100"],
            "delta_over_same_encoder": decision["best_memory_delta_over_same_encoder"],
        },
        "next_human_conditioned_identity_stage_authorized": decision["next_human_conditioned_identity_stage_authorized"],
        "next_association_stage_authorized": False,
        "sam3_run": False,
        "mot_evaluation_run": False,
        "training_run": False,
        "video_encoder_status": manifest.get("video_encoder"),
        "stop_rule": "No automatic SAM3/MOT/association/training work after answering the identity representation question.",
        "finalized_at_utc": now,
    }
    (output_dir / "FINAL_RESULT.json").write_text(json.dumps(final_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage_status = {
        **final_goal,
        "status": "COMPLETE",
        "decision": decision["decision"],
        "next_human_conditioned_identity_stage_authorized": decision["next_human_conditioned_identity_stage_authorized"],
        "results_complete": True,
        "finalized_at_utc": now,
    }
    (output_dir / "stage_status.json").write_text(json.dumps(stage_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    osnet_val = aggregate["encoder_benchmark"][ENCODER_OSNET]["val"]
    clip_val = aggregate["encoder_benchmark"][ENCODER_CLIP]["val"]
    best_name = decision["best_memory_variant"].split(":", 1)[1]
    best_encoder = decision["best_memory_variant"].split(":", 1)[0]
    best_memory_val = aggregate["memory_baselines"][best_encoder]["val"]["variants"][best_name]
    report = f'''FINAL GOAL:
{GOAL}

CENTRAL QUESTION:
“{QUESTION_ZH}”

# InterMOT N72R17 — Identity Representation Research Phase

## Scientific question

**{QUESTION_EN}**

N72R17 inherits the N72R16 protocol byte-for-byte: the earliest score-blind eligible human anchor, future gaps through H100, and the same-frame hard negative defined as the most similar competing identity. The primary endpoint remains `{PRIMARY}`.

## Scope and frozen controls

- No SAM3 inference, MOT association, Hungarian redesign, TrackEval, requery, LoRA, identity-decoder training, or full MOT evaluation was run.
- OSNet x1.0 / Market1501 is the reused N72R16 baseline embedding store.
- The new encoder is frozen public `timm/vit_base_patch32_clip_224.openai`, used as a generic CLIP visual representation. It was not fine-tuned for DanceTrack or ReID.
- The public OPA067/ReID reference pipeline was reviewed, but no compatible trained ReID checkpoint was available; this report must not be read as a CLIP-ReID fine-tuning result.
- Memory-3/5 and EMA/attention are offline diagnostics using future same-identity GT crops causally after each score; they are not deployable online memory claims.
- Video identity encoder: `{manifest.get('video_encoder', {}).get('status', 'UNKNOWN')}`. Reason: {manifest.get('video_encoder', {}).get('reason', 'not recorded')}.
- Only embeddings, metadata, JSONL records, and reports were stored; no crops were written.

## Stage A — encoder benchmark

{metric_table(aggregate, 'encoder_benchmark')}

## Stage B — memory diagnostics

{metric_table(aggregate, 'memory_baselines')}

## Required secondary analysis

The validation single-anchor time-gap breakdown is included below for both frozen encoders, with the best measured memory diagnostic shown afterward.

### OSNet single anchor — validation

{secondary_table(osnet_val)}

### OpenAI CLIP ViT-B/32 — validation

{secondary_table(clip_val)}

### Best memory diagnostic — `{best_encoder}:{best_name}` validation

{secondary_table(best_memory_val)}

## Decision

`{decision['decision']}`

{decision['rationale']}

- Validation OSNet H100 hard-negative win rate: **{pct(decision['osnet_h100'])}**.
- Validation CLIP H100 hard-negative win rate: **{pct(decision['clip_h100'])}**; delta over OSNet: **{pct(decision['clip_delta_over_osnet'])}**.
- Best memory diagnostic: **{decision['best_memory_variant']}**, H100 **{pct(decision['best_memory_h100'])}**, delta over its same-encoder single anchor **{pct(decision['best_memory_delta_over_same_encoder'])}**.
- Predeclared gates: encoder-limited requires CLIP H100 ≥50%, improvement ≥10 percentage points, and sequence-cluster CI lower >40%; memory-promising requires the analogous memory improvement ≥5 points plus H100 ≥50% and CI lower >40%.

## Authorization and stop rule

- `NEXT_ASSOCIATION_STAGE_AUTHORIZED`: **false**.
- `NEXT_HUMAN_CONDITIONED_IDENTITY_STAGE_AUTHORIZED`: **{str(decision['next_human_conditioned_identity_stage_authorized']).lower()}**.
- N72R17 stops here. No downstream SAM3/MOT/association or Stage C training starts automatically.

## Provenance

- Asset manifest: `{args.asset_manifest}`
- Aggregate: `{args.aggregate}`
- Protocol manifest: `{aggregate['protocol']}`
- Frozen Final Goal: `{output_dir / 'FINAL_GOAL.json'}`
- Final result: `{output_dir / 'FINAL_RESULT.json'}`
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(final_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
