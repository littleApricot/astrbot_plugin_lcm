"""Pure helpers for comparing and validating effective conversation history."""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque


def ordered_history_nodes(nodes, inactive_rounds=None):
    """Pair historical tools by protocol IDs, independently of edit ownership.

    Revisions may belong to a later run than the call they answer. Keep every
    archive node exactly once; incomplete executions become historical data.
    The returned wrappers are detached from the immutable archive records.
    """
    nodes = [dict(node) for node in nodes]
    inactive_rounds = inactive_rounds or {}
    pending, groups, results = defaultdict(deque), {}, {}
    for index, node in enumerate(nodes):
        message = node.get("payload", {}).get("message", {})
        calls = message.get("tool_calls") or []
        is_call = message.get("role") == "assistant" and bool(calls)
        is_result = message.get("role") == "tool"
        if not (is_call or is_result):
            continue
        reason = inactive_rounds.get(node.get("round_id"))
        if reason:
            node["_tool_record_reason"] = reason
            continue
        if is_call:
            ids = [call.get("id") if isinstance(call, dict) else None for call in calls]
            groups[index] = [None] * len(ids)
            if not all(isinstance(ident, str) and ident for ident in ids) or len(
                set(ids)
            ) != len(ids):
                node["_tool_record_reason"] = "invalid_call_ids"
                continue
            for position, ident in enumerate(ids):
                pending[str(ident)].append((index, position))
        elif is_result:
            ident = str(message.get("tool_call_id") or "")
            if pending[ident]:
                call_index, position = pending[ident].popleft()
                groups[call_index][position] = index
                results[index] = call_index
            else:
                node["_tool_record_reason"] = "unpaired_result"

    complete = set()
    for index, matches in groups.items():
        if all(match is not None for match in matches):
            complete.add(index)
        else:
            nodes[index].setdefault("_tool_record_reason", "incomplete_calls")
            for match in matches:
                if match is not None:
                    nodes[match]["_tool_record_reason"] = "incomplete_calls"
    ordered = []
    for index, node in enumerate(nodes):
        if results.get(index) in complete:
            continue
        ordered.append(node)
        if index in complete:
            ordered.extend(nodes[match] for match in groups[index])
    return ordered


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
