"""A tiny fake LSP server used by contract tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded or ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length") or "0")
    if length <= 0:
        return None
    payload = sys.stdin.buffer.read(length)
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def _send_message(payload):
    encoded = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _publish_diagnostics(uri: str):
    path = Path(uri.removeprefix("file://"))
    message = f"fake diagnostic for {path.name}"
    _send_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 4},
                        },
                        "severity": 2,
                        "source": "fake-lsp",
                        "code": "FAKE001",
                        "message": message,
                    }
                ],
            },
        }
    )


def main():
    should_exit = False
    while not should_exit:
        message = _read_message()
        if message is None:
            break
        request_id = message.get("id")
        method = message.get("method")
        params = dict(message.get("params") or {})

        if method == "initialize":
            _send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "capabilities": {
                            "definitionProvider": True,
                            "referencesProvider": True,
                            "documentSymbolProvider": True,
                            "workspaceSymbolProvider": True,
                            "textDocumentSync": 2,
                        }
                    },
                }
            )
        elif method == "initialized":
            continue
        elif method == "textDocument/didOpen":
            text_document = dict(params.get("textDocument") or {})
            uri = str(text_document.get("uri") or "")
            if uri:
                _publish_diagnostics(uri)
        elif method == "textDocument/didChange":
            continue
        elif method == "textDocument/didClose":
            continue
        elif method == "textDocument/definition":
            uri = str(dict(params.get("textDocument") or {}).get("uri") or "")
            _send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [
                        {
                            "uri": uri,
                            "range": {
                                "start": {"line": 0, "character": 4},
                                "end": {"line": 0, "character": 8},
                            },
                        }
                    ],
                }
            )
        elif method == "textDocument/references":
            uri = str(dict(params.get("textDocument") or {}).get("uri") or "")
            _send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [
                        {
                            "uri": uri,
                            "range": {
                                "start": {"line": 0, "character": 4},
                                "end": {"line": 0, "character": 8},
                            },
                        },
                        {
                            "uri": uri,
                            "range": {
                                "start": {"line": 2, "character": 0},
                                "end": {"line": 2, "character": 4},
                            },
                        },
                    ],
                }
            )
        elif method == "textDocument/documentSymbol":
            uri = str(dict(params.get("textDocument") or {}).get("uri") or "")
            _send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [
                        {
                            "name": "demo_function",
                            "kind": 12,
                            "detail": "fake detail",
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 1, "character": 0},
                            },
                            "selectionRange": {
                                "start": {"line": 0, "character": 4},
                                "end": {"line": 0, "character": 17},
                            },
                            "children": [],
                        }
                    ],
                }
            )
        elif method == "workspace/symbol":
            root_uri = str(params.get("query") or "")
            query = str(params.get("query") or "")
            fake_uri = Path.cwd().joinpath("sample.py").as_uri()
            _send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [
                        {
                            "name": query or "demo_symbol",
                            "kind": 12,
                            "containerName": "workspace",
                            "location": {
                                "uri": fake_uri,
                                "range": {
                                    "start": {"line": 0, "character": 0},
                                    "end": {"line": 0, "character": 10},
                                },
                            },
                        }
                    ],
                }
            )
        elif method == "shutdown":
            _send_message({"jsonrpc": "2.0", "id": request_id, "result": None})
        elif method == "exit":
            should_exit = True
        elif request_id is not None:
            _send_message({"jsonrpc": "2.0", "id": request_id, "result": None})


if __name__ == "__main__":
    main()
