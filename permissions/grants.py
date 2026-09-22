"""Runtime-owned, one-use approval receipts independent of policy implementations."""

from dataclasses import asdict
from copy import deepcopy
from threading import Lock

from .models import PermissionGrant


class PermissionGrantLedger:
    def __init__(self):
        self._records: dict[str, dict] = {}
        self._requests: set[str] = set()
        self._lock = Lock()

    def consume(self, grant: PermissionGrant) -> None:
        with self._lock:
            if not grant.request_id or not grant.grant_id:
                raise ValueError("Approval requires request and grant identities")
            if grant.grant_id in self._records or grant.request_id in self._requests:
                raise ValueError("Approval has already been consumed")
            self._records[grant.grant_id] = asdict(grant)
            self._requests.add(grant.request_id)

    def export_state(self) -> dict:
        with self._lock:
            return {"consumed": deepcopy(list(self._records.values()))}

    def restore_state(self, state) -> None:
        records = [PermissionGrant(**item) for item in dict(state or {}).get("consumed", [])]
        with self._lock:
            self._records = {item.grant_id: asdict(item) for item in records}
            self._requests = {item.request_id for item in records}
