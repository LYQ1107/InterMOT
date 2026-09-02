#!/usr/bin/env python3
"""Prepare the external real-human collection boundary without making events."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")
PLAN_PATH = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/N71/candidate_branch/window_plan.json"
)
PYTHON = "/home/lwr/anaconda3/envs/intermot/bin/python"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    windows = [dict(item) for item in json.loads(PLAN_PATH.read_text(encoding="utf-8")).get("windows", [])]
    if len(windows) < 4:
        raise RuntimeError("external collection queue requires at least four frozen structural windows")
    queue: list[dict[str, Any]] = []
    for index, item in enumerate(windows[:4], 1):
        queue.append({
            "queue_id": f"n72r1-collection-slot-{index:02d}",
            "sequence": str(item["sequence"]),
            "candidate_window_ref": f"six_window_export/windows/{item['window_id']}",
            "allowed_action_types": ["AUTHORITATIVE_CORRECT", "ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY"],
            "selection_basis": "structural runtime window and external annotation coverage planning only; no future GT/replay outcome",
            "status": "QUEUE_TEMPLATE_NOT_EVENT",
            "is_event": False,
            "real_human": False,
            "synthetic_fixture": False,
            "requires_external_ui_submission": True,
        })
    queue_path = N72R1_ROOT / "human_events" / "smoke_queue.json"
    atomic_json(queue_path, {
        "schema_version": "N72R1_EXTERNAL_COLLECTION_QUEUE_V1",
        "status": "QUEUE_ONLY_NOT_EVENT_TAPE",
        "queue_item_count": len(queue),
        "items": queue,
        "real_human_event_count": 0,
        "synthetic_from_gt_event_count": 0,
        "runtime_future_gt_used": False,
        "note": "These slots are not events and contain no public IDs, boxes, clicks, masks, or labels.",
    })
    ui_guide = f"""# N72R1 external human collection UI

当前 real-human tape 数量为 **0**。下面命令只启动外部提交入口，不会生成事件；只有标注者在浏览器/标注器中直接确认并提交，服务器才会把记录写入 `real_human_events.jsonl`。

## 启动

```bash
cd {ROOT}
PYTHONPATH={ROOT} {PYTHON} ui/n72r1_human_ui.py \\
  --host 127.0.0.1 --port 8762 \\
  --raw-root {N72R1_ROOT}/human_events/raw_namespace \\
  --event-root {N72R1_ROOT}/human_events/validated_namespace \\
  --annotator-id '<your-annotator-account>'
```

打开 `http://127.0.0.1:8762/`。客户端必须 POST 完整的 V2 JSON 到 `/api/events`；服务器先保存请求原始字节，再验证并追加 hash-chain 事件。浏览器提交前必须由人直接确认 action、event frame 和输入区域。

## 人工输入边界

- `AUTHORITATIVE_CORRECT`、`RECOVER_IDENTITY` 和 `ADD_NEW_IDENTITY` 使用人工直接提供的 BOX/CLICK/CONFIRMED_MASK；`ADD_NEW_IDENTITY` 不允许用户填写 public ID，ID 由 allocator 在运行时产生。
- `AUTHORITATIVE_REASSIGN`、`ATOMIC_ID_SWAP` 和 `AUTHORITATIVE_DELETE` 使用人工直接选择的已有 public ID；系统不从 GT、候选排序或数字 ID 推断身份。
- confirmed mask 必须是 lossless PNG/NPZ/RLE，且有独立 payload reference、shape 和 SHA-256；machine candidate mask 不能改名为 human mask。
- 必须同时提交 frame hash、candidate tape reference、prefix `[0,event_frame-1]`、H20/H50/H100 future ranges、session/annotator/timestamp；`runtime_future_gt_used` 必须为 `false`。
- N37/N39/N41/N42/N70/N71 的 `simulated_from_gt` 记录不是本 tape，不能导入、改名或当作历史点击。

## CPU-only 验证

```bash
PYTHONPATH={ROOT} {PYTHON} scripts/n72r1_validate_real_human_tape.py \\
  --event-root {N72R1_ROOT}/human_events/validated_namespace \\
  --raw-root {N72R1_ROOT}/human_events/raw_namespace \\
  --candidate-root {N72R1_ROOT}/six_window_export \\
  --report {N72R1_ROOT}/human_events/real_human_tape_audit.json
```

空 tape 的返回状态是 `PASS_EMPTY_REAL_TAPE`，不是实验成功。验证通过后，仍需单独完成 real full-loop、future replay 和严格 gate；本轮不会自动启动这些下游任务。
"""
    write_text(N72R1_ROOT / "ui" / "UI_GUIDE.md", ui_guide)
    morning = f"""# N72R1 morning actions

1. Inspect the read-only structural evidence: `{N72R1_ROOT}/six_window_export/integrity_audit.json` and `{N72R1_ROOT}/status/stage_16_status.json`.
2. Start the local-only external UI using `ui/UI_GUIDE.md`; do not fill forms from GT or simulated records.
3. Submit only direct human annotations. Preserve raw request bytes and any confirmed mask payload in the separate raw namespace.
4. Run the CPU-only validator command in `human_events/validation_command.txt`. `PASS_EMPTY_REAL_TAPE` means no input was collected.
5. Do not run replay, calibration, LoRA, or efficacy scoring until a non-empty schema-valid real-human tape and a public-authority bridge have both been independently audited.

The current N72R1 state is real-human-event count `0`; the queue is planning metadata, not tape.
"""
    write_text(N72R1_ROOT / "MORNING_ACTIONS.md", morning)
    validation_command = f"PYTHONPATH={ROOT} {PYTHON} scripts/n72r1_validate_real_human_tape.py --event-root {N72R1_ROOT}/human_events/validated_namespace --raw-root {N72R1_ROOT}/human_events/raw_namespace --candidate-root {N72R1_ROOT}/six_window_export --report {N72R1_ROOT}/human_events/real_human_tape_audit.json\n"
    replay_command = "NOT_READY: real-human tape, explicit same-run public authority mapping, and full-loop audit are required before replay; no replay command is authorized by N72R1 Stage 17.\n"
    write_text(N72R1_ROOT / "human_events" / "validation_command.txt", validation_command)
    write_text(N72R1_ROOT / "human_events" / "replay_readiness_command.txt", replay_command)
    status = {
        "schema_version": "N72R1_STAGE_STATUS_V1",
        "stage": "N72R1-17",
        "status": "PASS_REAL_HUMAN_COLLECTION_PREPARATION_REAL_COUNT_ZERO",
        "queue_path": str(queue_path),
        "queue_item_count": len(queue),
        "queue_is_event_tape": False,
        "real_human_event_count": 0,
        "synthetic_fixture_count": 0,
        "interaction_source": "NO_EVENTS_COLLECTED",
        "ui_ready": True,
        "validator_ready": True,
        "replay_started": False,
        "training_started": False,
        "runtime_future_gt_used": False,
        "plan_sha256": sha256(PLAN_PATH),
        "outputs": {
            "ui_guide": str(N72R1_ROOT / "ui" / "UI_GUIDE.md"),
            "morning_actions": str(N72R1_ROOT / "MORNING_ACTIONS.md"),
            "validation_command": str(N72R1_ROOT / "human_events" / "validation_command.txt"),
            "replay_readiness_command": str(N72R1_ROOT / "human_events" / "replay_readiness_command.txt"),
        },
        "next_minimum_action": "Collect external direct human annotations through the server UI; do not relabel simulated_from_gt artifacts.",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(N72R1_ROOT / "status" / "stage_17_status.json", status)
    print(json.dumps({"status": status["status"], "queue_items": len(queue), "real_human_event_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
