from executor import DefaultAgentExecutor


class _ProviderFailure(Exception):
    status_code = 403
    code = "request_blocked"
    request_id = None

    def __init__(self) -> None:
        super().__init__("Your request was blocked.")
        self.response = type(
            "Response",
            (),
            {
                "headers": {
                    "x-oneapi-request-id": "request-123",
                    "cf-ray": "edge-456",
                    "authorization": "must-not-be-exported",
                }
            },
        )()


def test_executor_error_payload_preserves_safe_provider_diagnostics() -> None:
    payload = DefaultAgentExecutor._error_payload(_ProviderFailure())

    assert payload == {
        "error_type": "_ProviderFailure",
        "error_message": "Your request was blocked.",
        "status_code": 403,
        "error_code": "request_blocked",
        "request_id": "request-123",
        "edge_trace_id": "edge-456",
    }
