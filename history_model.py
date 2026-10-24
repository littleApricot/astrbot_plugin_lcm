"""Pure helpers for comparing and validating effective conversation history."""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque


def history_key(message):
    """Compare native serializer aliases without adding fields to model messages."""

    def normalize(value):
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in sorted(value.items())
                if key != "_no_save" and item not in (None, "", [], {})
            }
        return value

    result = copy.deepcopy(message)
    content = result.get("content")
    if isinstance(content, list) and all(
        isinstance(part, dict)
        and part.get("type") == "text"
        and set(part) <= {"type", "text", "_no_save"}
        for part in content
    ):
        result["content"] = "".join(
            str(part.get("text", "")) for part in content if not part.get("_no_save")
        )
    return json.dumps(
        normalize(result), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def safe_history(messages):
    """Return a detached, savable history with complete tool call/result pairs."""
    saved = []
    for value in messages:
        if not isinstance(value, dict):
            raise ValueError("history_message_requires_object")
        if value.get("role") == "_checkpoint" or value.get("_no_save"):
            continue
        message = copy.deepcopy(value)
        message.pop("_no_save", None)
        if isinstance(message.get("content"), list):
            message["content"] = [
                {key: val for key, val in part.items() if key != "_no_save"}
                if isinstance(part, dict)
                else part
                for part in message["content"]
                if not isinstance(part, dict) or not part.get("_no_save")
            ]
        saved.append(message)

    pending = defaultdict(deque)
    paired_calls, paired_results = set(), set()
    for index, message in enumerate(saved):
        if message.get("role") == "assistant":
            for position, call in enumerate(message.get("tool_calls") or []):
                ident = call.get("id") if isinstance(call, dict) else None
                if ident:
                    pending[str(ident)].append((index, position))
        elif message.get("role") == "tool":
            ident = str(message.get("tool_call_id") or "")
            if pending[ident]:
                paired_calls.add(pending[ident].popleft())
                paired_results.add(index)

    result = []
    for index, message in enumerate(saved):
        if message.get("role") == "tool" and index not in paired_results:
            continue
        calls = message.get("tool_calls")
        if message.get("role") == "assistant" and calls:
            kept = [
                call
                for position, call in enumerate(calls)
                if (index, position) in paired_calls
            ]
            if kept:
                message["tool_calls"] = kept
            else:
                message.pop("tool_calls", None)
                if not message.get("content"):
                    continue
        result.append(message)
    return result
