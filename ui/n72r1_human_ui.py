#!/usr/bin/env python3
"""Minimal standard-library UI/server adapter for real N72R1 events.

This is an ingestion boundary, not a simulator.  A browser submission remains
``ui_submission`` until the server validates it, stores the exact request
bytes, adds the server session nonce, and finalizes it as ``real_human``.
No GT reader, event simulator, candidate selector, or public-ID inference is
imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping

from sam3_intermot.interaction.real_human_v2 import (
    finalize_ui_submission,
    new_server_session,
    validate_real_human_event_v2,
)
from sam3_intermot.provenance.append_only import AppendOnlyJSONL
from sam3_intermot.provenance.path_safety import resolve_within_root


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:32]


def schema_document() -> dict[str, Any]:
    return {
        "schema_version": "N72R1_UI_INGESTION_V1",
        "submission_source": "external_browser_or_annotation_client",
        "required_top_level": [
            "event_id", "sequence", "event_frame", "action_type", "human_confirmed",
            "session_id", "timestamp", "frame_hash_sha256", "candidate_tape_ref",
            "prefix_range", "future_ranges", "human_input",
        ],
        "human_input_kinds": ["BOX", "CLICK", "CONFIRMED_MASK", "ID_SELECTION"],
        "public_id_rule": "existing IDs are selected directly by the human; ADD_NEW_IDENTITY is allocated by the server",
        "finalization_rule": "only finalize_ui_submission with a server session can emit interaction_source=real_human",
        "raw_preservation": "store exact request bytes and lossless mask bytes in a separate raw namespace before validation",
        "runtime_future_gt_used": False,
        "synthetic_fixture_allowed": False,
    }


class N72R1IngestionService:
    """Validate and append one external submission under separate roots."""

    def __init__(self, *, raw_root: str | Path, event_root: str | Path, annotator_id: str, ui_version: str = "n72r1-ui-v1") -> None:
        self.raw_root = Path(raw_root).resolve()
        self.event_root = Path(event_root).resolve()
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.event_root.mkdir(parents=True, exist_ok=True)
        self.session = new_server_session(annotator_id, ui_version=ui_version)
        self.events = AppendOnlyJSONL(self.event_root / "real_human_events.jsonl", root=self.event_root, key_field="event_id")

    def _persist_exact_request(self, body: bytes) -> tuple[Path, str]:
        digest = hashlib.sha256(body).hexdigest()
        path = resolve_within_root(Path("requests") / f"{digest}.json", self.raw_root)
        if path.exists() and path.read_bytes() != body:
            raise ValueError("raw request digest collision with different bytes")
        if not path.exists():
            _atomic_bytes(path, body)
        return path, digest

    def submit_json_bytes(self, body: bytes) -> dict[str, Any]:
        """Persist exact body, then finalize and append only a valid event."""

        request_path, request_digest = self._persist_exact_request(body)
        try:
            record = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {"status": "REJECTED", "reason": f"invalid JSON: {exc}", "raw_request": str(request_path), "raw_request_sha256": request_digest}
        if not isinstance(record, dict):
            return {"status": "REJECTED", "reason": "submission root must be an object", "raw_request": str(request_path), "raw_request_sha256": request_digest}
        record = dict(record)
        human_input = dict(record.get("human_input") or {})
        # The server, not the client, defines which exact bytes are the raw
        # submission payload.  Existing client references are not trusted.
        relative_request = str(request_path.relative_to(self.raw_root))
        human_input["raw_payload_ref"] = relative_request
        human_input["raw_payload_sha256"] = request_digest
        record["human_input"] = human_input
        record["interaction_source"] = "ui_submission"
        # The active server session is authoritative for the annotator
        # provenance.  The browser still supplies the session ID, but cannot
        # forge a different annotator hash.
        record["annotator_id_hash"] = self.session["annotator_id_hash"]
        record["runtime_future_gt_used"] = False
        audit = validate_real_human_event_v2(record)
        if not audit["valid"]:
            return {"status": "REJECTED", "validation": audit, "raw_request": str(request_path), "raw_request_sha256": request_digest}
        try:
            finalized = finalize_ui_submission(record, self.session)
            appended = self.events.append(finalized)
        except Exception as exc:
            return {"status": "REJECTED", "reason": f"server finalization failed: {type(exc).__name__}: {exc}", "raw_request": str(request_path), "raw_request_sha256": request_digest}
        return {
            "status": "ACCEPTED_REAL_HUMAN_EVENT",
            "event_id": appended.get("event_id"),
            "interaction_source": appended.get("interaction_source"),
            "server_generated_real_human": appended.get("server_generated_real_human"),
            "raw_request": str(request_path),
            "raw_request_sha256": request_digest,
            "runtime_future_gt_used": False,
        }


def render_form() -> str:
    """Return a dependency-free form; the form does not create fake events."""

    return """<!doctype html>
<meta charset="utf-8"><title>N72R1 Human Event Ingestion</title>
<h1>N72R1 external human event</h1>
<p>Submit only a direct human annotation. No GT, simulator, or inferred public ID is accepted.</p>
<p>The client must POST the complete JSON contract to <code>/api/events</code>.</p>
<pre>event_frame -> current spatial correction -> server memory write -> future starts at event+1</pre>
"""


class _Handler(BaseHTTPRequestHandler):
    service: N72R1IngestionService | None = None

    def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            body = render_form().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self._write_json(200, {"status": "READY", "runtime_future_gt_used": False})
        else:
            self._write_json(404, {"status": "NOT_FOUND"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/events" or self.service is None:
            self._write_json(404, {"status": "NOT_FOUND"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 4 * 1024 * 1024:
            self._write_json(413, {"status": "REJECTED", "reason": "request size must be 1..4 MiB"})
            return
        result = self.service.submit_json_bytes(self.rfile.read(length))
        self._write_json(200 if result.get("status", "").startswith("ACCEPTED") else 400, result)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(service: N72R1IngestionService, *, host: str = "127.0.0.1", port: int = 8762) -> None:
    _Handler.service = service
    server = HTTPServer((host, int(port)), _Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="N72R1 real-human ingestion UI")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8762)
    args = parser.parse_args()
    run_server(N72R1IngestionService(raw_root=args.raw_root, event_root=args.event_root, annotator_id=args.annotator_id), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
