import copy
import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.providers import create_codec


class _RawPayloadTextPart:
    def __init__(self, text: str, *, raw_payload: dict | None = None):
        self.text = text
        self.thought = False
        self._raw_payload = raw_payload or {"text": text}

    def to_dict(self):
        return copy.deepcopy(self._raw_payload)


class _RawPayloadFunctionCallPart:
    def __init__(
        self,
        *,
        function_id,
        name: str,
        args: dict,
        thought_signature,
        raw_payload: dict,
    ):
        self.function_call = SimpleNamespace(id=function_id, name=name, args=args)
        self.thought_signature = thought_signature
        self._raw_payload = raw_payload

    def to_dict(self):
        return copy.deepcopy(self._raw_payload)


class TestGoogleNativeStreamReplay(unittest.TestCase):
    def test_build_tool_result_omits_unknown_id(self):
        codec = create_codec("google_native")
        payload = codec.build_tool_result("531441", "unknown", "calculator")
        function_response = payload["parts"][0]["function_response"]
        self.assertEqual(function_response["name"], "calculator")
        self.assertNotIn("id", function_response)

    def test_stream_tool_calls_normalize_real_sdk_model_dump_payload(self):
        try:
            from google.genai import types as gt
        except Exception as exc:
            self.skipTest(f"google-genai unavailable: {exc}")

        codec = create_codec("google_native")
        signed_call = gt.Part(
            functionCall=gt.FunctionCall(id="call_1", name="calculator", args={"expression": "3**12"}),
            thought_signature=b"sig-raw",
        )
        raw_stream = [
            SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(role="model", parts=[signed_call]),
                    )
                ],
            )
        ]

        events = list(codec.stream_events(raw_stream, tools=True))

        assistant_message = events[-1]["assistant_items"][0]
        self.assertEqual(
            assistant_message,
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "call_1",
                            "name": "calculator",
                            "args": {"expression": "3**12"},
                        },
                        "thought_signature": "c2lnLXJhdw==",
                    }
                ],
            },
        )

    def test_stream_tool_calls_prefer_raw_provider_payload_over_accessor_id(self):
        codec = create_codec("google_native")
        signed_call = _RawPayloadFunctionCallPart(
            function_id="synthetic-stream-id",
            name="calculator",
            args={"expression": "3**12"},
            thought_signature="sig-raw",
            raw_payload={
                "functionCall": {
                    "name": "calculator",
                    "args": {"expression": "3**12"},
                },
                "thoughtSignature": "sig-raw",
            },
        )
        raw_stream = [
            SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(role="model", parts=[signed_call]),
                    )
                ],
            )
        ]

        events = list(codec.stream_events(raw_stream, tools=True))

        tool_event = events[-1]
        self.assertEqual(tool_event["type"], "tool_calls")
        self.assertIsNone(tool_event["tool_calls"][0]["id"])
        assistant_message = tool_event["assistant_items"][0]
        self.assertEqual(assistant_message["role"], "model")
        self.assertEqual(assistant_message["parts"][0]["function_call"]["name"], "calculator")
        self.assertNotIn("id", assistant_message["parts"][0]["function_call"])
        self.assertEqual(assistant_message["parts"][0]["thought_signature"], "sig-raw")

    def test_stream_tool_calls_drop_empty_text_parts_before_signed_function_call(self):
        codec = create_codec("google_native")
        empty_text = _RawPayloadTextPart("", raw_payload={"text": ""})
        signed_call = _RawPayloadFunctionCallPart(
            function_id="synthetic-stream-id",
            name="translate_tool",
            args={"text": "你是谁，在哪里", "target_lang": "English"},
            thought_signature="sig-translate",
            raw_payload={
                "functionCall": {
                    "name": "translate_tool",
                    "args": {"text": "你是谁，在哪里", "target_lang": "English"},
                },
                "thoughtSignature": "sig-translate",
            },
        )
        raw_stream = [
            SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(role="model", parts=[empty_text, signed_call]),
                    )
                ],
            )
        ]

        events = list(codec.stream_events(raw_stream, tools=True))

        assistant_message = events[-1]["assistant_items"][0]
        self.assertEqual(len(assistant_message["parts"]), 1)
        self.assertEqual(assistant_message["parts"][0]["function_call"]["name"], "translate_tool")
        self.assertNotIn("id", assistant_message["parts"][0]["function_call"])

    def test_stream_tool_calls_deduplicate_repeated_function_call_chunks(self):
        codec = create_codec("google_native")
        signed_call = _RawPayloadFunctionCallPart(
            function_id="synthetic-stream-id",
            name="calculator",
            args={"expression": "3**12"},
            thought_signature="sig-raw",
            raw_payload={
                "functionCall": {
                    "name": "calculator",
                    "args": {"expression": "3**12"},
                },
                "thoughtSignature": "sig-raw",
            },
        )
        chunk = SimpleNamespace(
            text=None,
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(role="model", parts=[signed_call]),
                )
            ],
        )
        events = list(codec.stream_events([chunk, chunk], tools=True))

        tool_event = events[-1]
        self.assertEqual(len(tool_event["tool_calls"]), 1)
        assistant_message = tool_event["assistant_items"][0]
        self.assertEqual(len(assistant_message["parts"]), 1)
        self.assertEqual(assistant_message["parts"][0]["function_call"]["name"], "calculator")

    def test_stream_parallel_tool_calls_preserve_provider_part_shape(self):
        codec = create_codec("google_native")
        first_call = _RawPayloadFunctionCallPart(
            function_id="synthetic-first-id",
            name="translate_tool",
            args={"text": "你是谁，在哪里", "target_lang": "English"},
            thought_signature="sig-first",
            raw_payload={
                "functionCall": {
                    "name": "translate_tool",
                    "args": {"text": "你是谁，在哪里", "target_lang": "English"},
                },
                "thoughtSignature": "sig-first",
            },
        )
        second_call = _RawPayloadFunctionCallPart(
            function_id="real-second-id",
            name="calculator",
            args={"expression": "3**22"},
            thought_signature=None,
            raw_payload={
                "functionCall": {
                    "id": "real-second-id",
                    "name": "calculator",
                    "args": {"expression": "3**22"},
                }
            },
        )
        raw_stream = [
            SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(role="model", parts=[first_call, second_call]),
                    )
                ],
            )
        ]

        events = list(codec.stream_events(raw_stream, tools=True))

        tool_event = events[-1]
        self.assertEqual(
            [call["id"] for call in tool_event["tool_calls"]],
            [None, "real-second-id"],
        )
        assistant_message = tool_event["assistant_items"][0]
        self.assertEqual(len(assistant_message["parts"]), 2)
        self.assertNotIn("id", assistant_message["parts"][0]["function_call"])
        self.assertEqual(assistant_message["parts"][0]["thought_signature"], "sig-first")
        self.assertEqual(
            assistant_message["parts"][1]["function_call"]["id"],
            "real-second-id",
        )

    def test_non_stream_get_tool_calls_keeps_missing_id_as_none(self):
        codec = create_codec("google_native")
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        role="model",
                        parts=[
                            {
                                "function_call": {
                                    "name": "calculator",
                                    "args": {"expression": "3**12"},
                                }
                            }
                        ],
                    )
                )
            ]
        )
        self.assertEqual(
            codec.get_tool_calls(response),
            [{"id": None, "name": "calculator", "arguments": {"expression": "3**12"}}],
        )


if __name__ == "__main__":
    unittest.main()
