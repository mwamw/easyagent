"""Minimal stdio LSP client used by EasyAgent code intelligence."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, unquote, urlparse


def path_to_uri(path: str) -> str:
    return Path(os.path.abspath(path)).as_uri()


def uri_to_path(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Only file:// URIs are supported, got: {uri}")
    return os.path.abspath(unquote(parsed.path))


class LSPClientError(RuntimeError):
    """Raised when the LSP client cannot fulfill a request."""


@dataclass(slots=True)
class _PendingResponse:
    event: threading.Event
    response: Optional[dict[str, Any]] = None
    error: Optional[BaseException] = None


class LSPClient:
    def __init__(
        self,
        *,
        server_command: list[str],
        workspace_root: str,
        env: Optional[dict[str, str]] = None,
        request_timeout_ms: int = 5000,
        diagnostics_wait_ms: int = 800,
    ):
        if not server_command:
            raise ValueError("server_command 不能为空。")
        self.server_command = list(server_command)
        self.workspace_root = os.path.abspath(workspace_root)
        self.request_timeout_ms = max(100, int(request_timeout_ms))
        self.diagnostics_wait_ms = max(100, int(diagnostics_wait_ms))
        merged_env = os.environ.copy()
        merged_env.update(dict(env or {}))
        self._process = subprocess.Popen(
            self.server_command,
            cwd=self.workspace_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise LSPClientError("启动语言服务器失败：stdio 不可用。")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._stderr = self._process.stderr
        self._write_lock = threading.RLock()
        self._pending: dict[int, _PendingResponse] = {}
        self._next_request_id = 1
        self._reader_stop = threading.Event()
        self._stderr_tail: list[str] = []
        self._diagnostics_by_uri: dict[str, list[dict[str, Any]]] = {}
        self._diagnostic_events: dict[str, threading.Event] = {}
        self._open_documents: dict[str, tuple[int, str]] = {}
        self._reader_thread = threading.Thread(target=self._reader_loop, name="easyagent-lsp-reader", daemon=True)
        self._stderr_thread = threading.Thread(target=self._drain_stderr_loop, name="easyagent-lsp-stderr", daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()
        self._initialize()

    def _initialize(self) -> None:
        root_uri = path_to_uri(self.workspace_root)
        response = self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "rootPath": self.workspace_root,
                "workspaceFolders": [
                    {
                        "uri": root_uri,
                        "name": os.path.basename(self.workspace_root) or self.workspace_root,
                    }
                ],
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {
                            "relatedInformation": True,
                            "tagSupport": {"valueSet": [1, 2]},
                        },
                        "definition": {"dynamicRegistration": False},
                        "references": {"dynamicRegistration": False},
                        "documentSymbol": {
                            "hierarchicalDocumentSymbolSupport": True,
                            "symbolKind": {"valueSet": list(range(1, 27))},
                        },
                    },
                    "workspace": {
                        "symbol": {
                            "dynamicRegistration": False,
                            "symbolKind": {"valueSet": list(range(1, 27))},
                        }
                    },
                },
            },
        )
        if "error" in response:
            raise LSPClientError(f"LSP initialize 失败: {response['error']}")
        self.notify("initialized", {})

    def _reader_loop(self) -> None:
        try:
            while not self._reader_stop.is_set():
                message = self._read_message()
                if message is None:
                    break
                self._handle_message(message)
        except Exception as exc:
            self._fail_pending(exc)

    def _drain_stderr_loop(self) -> None:
        if self._stderr is None:
            return
        try:
            while not self._reader_stop.is_set():
                line = self._stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if not decoded:
                    continue
                self._stderr_tail.append(decoded)
                if len(self._stderr_tail) > 50:
                    self._stderr_tail = self._stderr_tail[-50:]
        except Exception:
            return

    def _read_message(self) -> Optional[dict[str, Any]]:
        headers: dict[str, str] = {}
        while True:
            line = self._stdout.readline()
            if not line:
                return None
            if line in {b"\r\n", b"\n"}:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                break
            if ":" not in decoded:
                continue
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        content_length = int(headers.get("content-length") or "0")
        if content_length <= 0:
            return None
        payload = self._stdout.read(content_length)
        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))

    def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            request_id = int(message["id"])
            pending = self._pending.get(request_id)
            if pending is None:
                return
            pending.response = message
            pending.event.set()
            return
        method = str(message.get("method") or "")
        params = dict(message.get("params") or {})
        if method == "textDocument/publishDiagnostics":
            uri = str(params.get("uri") or "")
            diagnostics = list(params.get("diagnostics") or [])
            if uri:
                self._diagnostics_by_uri[uri] = diagnostics
                event = self._diagnostic_events.setdefault(uri, threading.Event())
                event.set()

    def _fail_pending(self, exc: BaseException) -> None:
        for pending in list(self._pending.values()):
            pending.error = exc
            pending.event.set()

    def _send_payload(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
        with self._write_lock:
            self._stdin.write(header)
            self._stdin.write(encoded)
            self._stdin.flush()

    def request(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        pending = _PendingResponse(event=threading.Event())
        self._pending[request_id] = pending
        self._send_payload(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params or {}),
            }
        )
        if not pending.event.wait(self.request_timeout_ms / 1000.0):
            self._pending.pop(request_id, None)
            raise LSPClientError(f"LSP 请求超时: {method}")
        self._pending.pop(request_id, None)
        if pending.error is not None:
            raise LSPClientError(f"LSP 通信失败: {pending.error}")
        return dict(pending.response or {})

    def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        self._send_payload(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": dict(params or {}),
            }
        )

    @staticmethod
    def _guess_language_id(path: str) -> str:
        suffix = Path(path).suffix.lower()
        mapping = {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".json": "json",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
        }
        return mapping.get(suffix, "plaintext")

    def open_document(self, path: str) -> str:
        resolved = os.path.abspath(path)
        uri = path_to_uri(resolved)
        with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        version, previous_text = self._open_documents.get(uri, (0, ""))
        if version == 0:
            self.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": self._guess_language_id(resolved),
                        "version": 1,
                        "text": text,
                    }
                },
            )
            self._open_documents[uri] = (1, text)
            return uri
        if previous_text != text:
            next_version = version + 1
            self.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": next_version},
                    "contentChanges": [{"text": text}],
                },
            )
            self._open_documents[uri] = (next_version, text)
        return uri

    def close_document(self, path: str) -> None:
        uri = path_to_uri(path)
        if uri not in self._open_documents:
            return
        self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
        self._open_documents.pop(uri, None)

    def definition(self, path: str, *, line_zero_based: int, column_zero_based: int) -> list[dict[str, Any]]:
        uri = self.open_document(path)
        response = self.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line_zero_based, "character": column_zero_based},
            },
        )
        result = response.get("result")
        if not result:
            return []
        if isinstance(result, list):
            return [dict(item) for item in result if item]
        if isinstance(result, dict):
            return [dict(result)]
        return []

    def references(
        self,
        path: str,
        *,
        line_zero_based: int,
        column_zero_based: int,
        include_declaration: bool,
    ) -> list[dict[str, Any]]:
        uri = self.open_document(path)
        response = self.request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line_zero_based, "character": column_zero_based},
                "context": {"includeDeclaration": bool(include_declaration)},
            },
        )
        result = response.get("result")
        if not result:
            return []
        return [dict(item) for item in list(result or []) if item]

    def document_symbols(self, path: str) -> list[dict[str, Any]]:
        uri = self.open_document(path)
        response = self.request(
            "textDocument/documentSymbol",
            {
                "textDocument": {"uri": uri},
            },
        )
        result = response.get("result")
        if not result:
            return []
        return [dict(item) for item in list(result or []) if item]

    def workspace_symbols(self, query: str) -> list[dict[str, Any]]:
        response = self.request(
            "workspace/symbol",
            {
                "query": str(query or ""),
            },
        )
        result = response.get("result")
        if not result:
            return []
        return [dict(item) for item in list(result or []) if item]

    def wait_for_diagnostics(self, path: str) -> list[dict[str, Any]]:
        uri = self.open_document(path)
        event = self._diagnostic_events.setdefault(uri, threading.Event())
        event.wait(self.diagnostics_wait_ms / 1000.0)
        return [dict(item) for item in list(self._diagnostics_by_uri.get(uri) or []) if item]

    def close(self) -> None:
        try:
            self.request("shutdown", {})
        except Exception:
            pass
        try:
            self.notify("exit", {})
        except Exception:
            pass
        self._reader_stop.set()
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except Exception:
            pass
        try:
            if self._process.stdout:
                self._process.stdout.close()
        except Exception:
            pass
        try:
            if self._process.stderr:
                self._process.stderr.close()
        except Exception:
            pass
        try:
            self._process.terminate()
        except Exception:
            pass
        try:
            self._process.wait(timeout=0.5)
        except Exception:
            pass


__all__ = ["LSPClient", "LSPClientError", "path_to_uri", "uri_to_path"]
