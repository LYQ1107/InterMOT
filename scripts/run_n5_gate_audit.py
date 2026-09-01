#!/usr/bin/env python
"""Audit N5-2 gate outputs: authority, invariants, GT access, rollback."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from sam3_intermot.evaluation.mot_export import validate_mot_file


ROOT = Path(".")
GATE = ROOT / "outputs/n5/gate"
SEQS = ["dancetrack0004", "dancetrack0007"]


def iou(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_mot(path):
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split(",")
        f = int(float(p[0]))
        tid = int(float(p[1]))
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        rows.setdefault(f, []).append((tid, np.asarray([x, y, x + w, y + h], float)))
    return rows


def audit_one(seq):
    run_dir = GATE / "runs" / seq
    events = []
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    post = read_mot(run_dir / "post_mot" / f"{seq}.txt")
    violations = []
    authority_ok = 0
    authority_total = 0
    box_cmds = {
        "ADD_NEW_IDENTITY",
        "RECOVER_IDENTITY",
        "AUTHORITATIVE_CORRECT",
    }
    for e in events:
        f = e["frame"]
        if e["action_type"] == "AUTHORITATIVE_DELETE":
            authority_total += 1
            box = e.get("authoritative_box")
            found = False
            for tid, b in post.get(f, []):
                if box is not None and iou(b, box) > 0.95:
                    found = True
                    break
            if not found:
                authority_ok += 1
            else:
                violations.append(f"{seq}:{f} DELETE row still present")
            continue
        if e["action_type"] not in box_cmds:
            continue
        authority_total += 1
        box = e.get("authoritative_box")
        if box is None:
            violations.append(f"{seq}:{f} missing authoritative box")
            continue
        found = False
        for tid, b in post.get(f, []):
            if iou(b, box) > 0.95:
                found = True
                break
        if found:
            authority_ok += 1
        else:
            violations.append(f"{seq}:{f} authority box not in post")
    mot_violations = validate_mot_file(run_dir / "post_mot" / f"{seq}.txt")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((run_dir / "gt_access_audit.json").read_text(encoding="utf-8"))
    return {
        "sequence": seq,
        "events": len(events),
        "accepted": summary["accepted_commands"],
        "authority_accuracy": authority_ok / authority_total if authority_total else 1.0,
        "authority_checked": authority_total,
        "violations": violations + mot_violations + summary.get("invariant_violations", []),
        "rolled_back": summary.get("rolled_back", []),
        "rejected": summary.get("rejected", []),
        "gt_audit_ok": audit["gt_read_before_prediction"] == 0
        and audit["gt_read_future"] == 0
        and audit["gt_used_for_model_decision"] == 0
        and audit["gt_used_for_scheduler"] == 0,
    }


def main():
    results = [audit_one(s) for s in SEQS]
    all_violations = []
    all_events = []
    for r in results:
        all_violations += [f"{r['sequence']}:{v}" for v in r["violations"]]
        ev_file = GATE / "runs" / r["sequence"] / "events.jsonl"
        all_events += ev_file.read_text(encoding="utf-8").splitlines()
    (GATE / "gate_audit.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (GATE / "invariant_violations.csv").write_text(
        "sequence,violation\n"
        + "".join(f"{r['sequence']},{v}\n" for r in results for v in r["violations"]),
        encoding="utf-8",
    )
    (GATE / "events.jsonl").write_text("\n".join(all_events) + "\n", encoding="utf-8")
    audit_ok = all(r["gt_audit_ok"] for r in results)
    no_viol = all(not r["violations"] for r in results)
    no_rollback = all(not r["rolled_back"] for r in results)
    authority = all(r["authority_accuracy"] >= 0.999 for r in results)
    gate_pass = audit_ok and no_viol and no_rollback and authority
    (GATE / "gate_result.json").write_text(
        json.dumps(
            {
                "verdict": "PASS" if gate_pass else "FAIL",
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": "PASS" if gate_pass else "FAIL", "results": results}, indent=2))
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
