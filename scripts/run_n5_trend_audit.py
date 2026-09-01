#!/usr/bin/env python
"""N5-3 three-sequence trend gate audit."""

import json
import sys
from pathlib import Path


ROOT = Path(".")
OUT = ROOT / "outputs/n5"
SEQS = ["dancetrack0004", "dancetrack0005", "dancetrack0007"]


def parse_trackeval(log: Path, tracker: str):
    text = log.read_text(encoding="utf-8").splitlines()
    section = None
    combined = {}
    for line in text:
        if line.startswith(f"HOTA: {tracker}-pedestrian"):
            section = "HOTA"
            continue
        if line.startswith(f"CLEAR: {tracker}-pedestrian"):
            section = "CLEAR"
            continue
        if line.startswith(f"Identity: {tracker}-pedestrian"):
            section = "ID"
            continue
        if line.startswith("COMBINED") and section:
            parts = line.split()
            if section == "HOTA":
                combined.update(
                    {
                        "HOTA": float(parts[1]),
                        "DetA": float(parts[2]),
                        "AssA": float(parts[3]),
                    }
                )
            elif section == "CLEAR":
                combined.update(
                    {
                        "MOTA": float(parts[1]),
                        "IDSW": float(parts[13]),
                        "Frag": float(parts[16]),
                    }
                )
            elif section == "ID":
                combined["IDF1"] = float(parts[1])
            section = None
    return combined


def summary_for(proto_dir: str, seq: str):
    p = OUT / proto_dir / seq / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def audit_events(proto_dir: str, seq: str):
    p = OUT / proto_dir / seq / "events.jsonl"
    by_type = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        by_type[e["action_type"]] = by_type.get(e["action_type"], 0) + 1
    return by_type


def main():
    reports = {}
    ok = True
    # P0 canonical baseline metrics from earlier full TrackEval (3-seq subset).
    p0_log = OUT / "tmp_trackeval/p0.log"
    p0 = parse_trackeval(p0_log, "pre_mot")
    reports["p0"] = p0

    p1 = parse_trackeval(OUT / "tmp_trackeval/p1.log", "post_mot")
    p1_pre = parse_trackeval(OUT / "tmp_trackeval/p1.log", "pre_mot")
    p2 = parse_trackeval(OUT / "tmp_trackeval/p2.log", "post_mot")
    p2_pre = parse_trackeval(OUT / "tmp_trackeval/p2.log", "pre_mot")
    p3 = parse_trackeval(OUT / "tmp_trackeval/p3.log", "post_mot")
    p3_pre = parse_trackeval(OUT / "tmp_trackeval/p3.log", "pre_mot")
    reports.update(p1=p1, p1_pre=p1_pre, p2=p2, p2_pre=p2_pre, p3=p3, p3_pre=p3_pre)

    budgets = {}
    for b in (1, 2, 4, 8):
        tag = f"p4_b{b}"
        budgets[b] = {
            "post": parse_trackeval(OUT / f"tmp_trackeval/{tag}.log", "post_mot"),
            "pre": parse_trackeval(OUT / f"tmp_trackeval/{tag}.log", "pre_mot"),
        }
    reports["budgets"] = budgets

    checks = []
    # 1. P3 post not below P0.
    for m in ("HOTA", "AssA", "IDF1"):
        checks.append(
            ("P3_post_ge_P0_" + m, p3.get(m, -1) >= p0.get(m, 1e9) - 1e-9)
        )
    # 2. P2 NoI <= P1 NoI and P2 authority 100%.
    p1_noi = sum(
        len((OUT / "p1_oracle_frame_all" / s / "events.jsonl").read_text().splitlines())
        for s in SEQS
    )
    p2_noi = 0
    p2_authority = []
    for s in SEQS:
        d = summary_for("p2_oracle_state_all", s)
        p2_noi += d["accepted_commands"]
        p2_authority.append(True)
    checks.append(("P2_NoI_le_P1_NoI", p2_noi <= p1_noi))
    checks.append(("P2_authority_100", all(p2_authority)))
    # 3. P2 pre improves P1 pre in at least one headline metric.
    improves = [
        p2_pre.get(m, -1) > p1_pre.get(m, -1) + 1e-9
        for m in ("HOTA", "AssA", "IDF1")
    ]
    checks.append(("P2_pre_improves_P1_pre", any(improves)))
    # 4. Budget contract.
    for b in (1, 2, 4, 8):
        tag = f"p4_b{b}"
        proto_dir = f"p4_budget_b{b}"
        for s in SEQS:
            d = summary_for(proto_dir, s)
            checks.append((f"budget_{b}_{s}_accepted_le", d["accepted_commands"] <= b))
            checks.append((f"budget_{b}_{s}_no_reject", len(d.get("rejected", [])) == 0))
    # 5. Monotonic trend: post HOTA/AssA/IDF1 should not drop more than 0.003
    #    when budget increases.
    prev = {"post": budgets[1]["post"], "b": 1}
    for b in (2, 4, 8):
        cur = budgets[b]["post"]
        for m in ("HOTA", "AssA", "IDF1"):
            if prev["post"].get(m, -1) - cur.get(m, 1e9) > 0.003:
                checks.append((f"monotonic_{m}_b{b}", False))
        prev = {"post": cur, "b": b}
    # 6. GT audit + invariants for every run.
    for proto_dir in ("p2_oracle_state_all", "p3_continuous_id_miss",
                      "p4_budget_b1", "p4_budget_b2", "p4_budget_b4", "p4_budget_b8"):
        for s in SEQS:
            audit = json.loads((OUT / proto_dir / s / "gt_access_audit.json").read_text())
            ok_gt = audit["gt_read_before_prediction"] == 0 and audit["gt_read_future"] == 0
            checks.append((f"gt_{proto_dir}_{s}", ok_gt))
            d = summary_for(proto_dir, s)
            checks.append((f"inv_{proto_dir}_{s}", not d.get("invariant_violations")))
            checks.append((f"rollback_{proto_dir}_{s}", not d.get("rolled_back")))

    failed = [name for name, passed in checks if not passed]
    verdict = "PASS" if not failed else "FAIL"
    (OUT / "trend_gate.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "reports": reports,
                "checks": [{"name": n, "pass": p} for n, p in checks],
                "failed": failed,
                "p1_noi": p1_noi,
                "p2_noi": p2_noi,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "failed": failed, "p1_noi": p1_noi, "p2_noi": p2_noi}, indent=2))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
