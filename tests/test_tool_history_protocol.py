"""Replay revised/interleaved archives through the native OpenAI serializer."""

import copy
import json

import httpx
import pytest
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from openai import AsyncOpenAI, BadRequestError

from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.history_model import ordered_history_nodes
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_effective_pipeline import MemoryConversationManager

owner = test_integration.owner


def call(*ids):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": ident,
                "type": "function",
                "function": {
                    "name": "get_self_prompt",
                    "arguments": "{}",
                },
            }
            for ident in ids
        ],
    }


def result(ident, text="old result"):
    return {"role": "tool", "tool_call_id": ident, "content": text}


def strict_protocol(messages):
    """Validate the outgoing HTTP body as strict tool-enabled APIs do."""
    pending = set()
    for message in messages:
        if message["role"] == "tool":
            ident = message.get("tool_call_id")
            if ident not in pending:
                return False
            pending.remove(ident)
        else:
            if pending:
                return False
            pending.update(item["id"] for item in message.get("tool_calls") or [])
    return not pending


def test_protocol_order_preserves_every_node_and_handles_partial_pairs():
    messages = [
        call("a", "b"),
        call("c"),
        result("b"),
        {"role": "assistant", "content": "parallel final"},
        result("c"),
        result("a"),
        call("a"),
        result("a", "reused id"),
        call("partial", "missing"),
        result("partial"),
        result("orphan"),
        call("failed"),
        result("failed"),
        call("", "invalid"),
    ]
    nodes = [
        {
            "id": str(i),
            "round_id": "bad" if i == 12 else "ok",
            "payload": {"message": msg},
        }
        for i, msg in enumerate(messages)
    ]
    before = copy.deepcopy(nodes)
    ordered = ordered_history_nodes(nodes, {"bad": "failed"})
    assert nodes == before
    assert sorted(int(node["id"]) for node in ordered) == list(range(len(nodes)))
    assert [int(node["id"]) for node in ordered[:8]] == [0, 5, 2, 1, 4, 3, 6, 7]
    assert all(node.get("_tool_record_reason") for node in ordered[8:])
    native = [
        node["payload"]["message"]
        for node in ordered
        if not node.get("_tool_record_reason")
    ]
    assert strict_protocol(native)


def revised_legacy(owner):
    before = [
        {"role": "user", "content": "old question"},
        call("memory-1"),
        result("memory-1"),
        {"role": "assistant", "content": "old final"},
    ]
    ids = [owner.store.append(owner.scope, msg, round_id="legacy") for msg in before]
    owner.store.sync_history(owner.scope, before, before)
    run = owner.store.begin(
        owner.scope, "edit-owner", owner.store.view(owner.scope), []
    )
    after = copy.deepcopy(before)
    after[2]["content"] = "revised result"
    owner.store.sync_history(
        owner.scope, before, after, run=run, internal=True, editable=ids
    )
    owner.store.finish(run)
    nodes = owner.store.view(owner.scope)["nodes"]
    assert next(n for n in nodes if n["kind"] == "revision")["round_id"] == run
    assert owner.store.node(owner.scope, ids[2])["payload"]["message"] == before[2]
    return nodes, after


@pytest.mark.asyncio
async def test_revised_legacy_pair_survives_native_http_serialization(owner):
    nodes, expected = revised_legacy(owner)
    # Reproduce the former run-based assembler before asserting the repair.
    runs = list(dict.fromkeys(node["round_id"] for node in nodes))
    broken = [
        node["payload"]["message"]
        for run in runs
        for node in nodes
        if node["round_id"] == run
    ]
    assert not strict_protocol(broken)
    fixed = owner.messages(nodes)
    assert fixed == expected
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        if not strict_protocol(body["messages"]):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "insufficient tool messages following tool_calls message",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "offline",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    provider = ProviderOpenAIOfficial(
        {
            "id": "offline",
            "type": "openai_chat_completion",
            "key": ["offline-marker"],
            "model": "offline",
            "api_base": "https://offline.invalid/v1",
        },
        {},
    )
    await provider.client.close()
    provider.client = AsyncOpenAI(
        api_key="offline-marker",
        base_url="https://offline.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    try:
        with pytest.raises(BadRequestError):
            await provider.text_chat(contexts=broken, prompt="current")
        response = await provider.text_chat(contexts=fixed, prompt="current")
        assert response.completion_text == "ok"
        assert len(bodies) == 2
        assert not strict_protocol(bodies[0]["messages"])
        assert strict_protocol(bodies[1]["messages"])
        assert any(
            msg.get("content") == "revised result" for msg in bodies[1]["messages"]
        )
    finally:
        await provider.client.close()


def test_incomplete_and_failed_tools_remain_readable_historical_data(owner):
    nodes = []
    for msg in [call("partial", "missing"), result("partial"), result("orphan")]:
        ident = owner.store.append(owner.scope, msg)
        nodes.append(owner.store.node(owner.scope, ident))
    run = owner.store.begin(owner.scope, "bad-owner", owner.store.view(owner.scope), [])
    for msg in [call("bad"), result("bad")]:
        ident = owner.store.append(owner.scope, msg, round_id=run)
        nodes.append(owner.store.node(owner.scope, ident))
    owner.store.finish(run, "failed")
    messages = owner.messages(nodes)
    assert len(messages) == len(nodes)
    assert strict_protocol(messages)
    assert all(msg["role"] == "user" for msg in messages)
    assert all(
        json.loads(msg["content"].split("\n", 1)[1]) == node["payload"]["message"]
        for msg, node in zip(messages, nodes)
    )


def test_history_sync_matches_projected_tool_images_to_original_ids(owner):
    import base64
    from types import SimpleNamespace

    from astrbot_plugin_lcm.tests.test_image_context import PNG

    original = "https://archive.invalid/tool.png"
    messages = [
        call("image-call"),
        {
            "role": "tool",
            "tool_call_id": "image-call",
            "content": [{"type": "image_url", "image_url": {"url": original}}],
        },
    ]
    ids = [owner.store.append(owner.scope, msg) for msg in messages]
    path = owner.media.root / "tool.png"
    path.write_bytes(base64.b64decode(PNG))
    with owner.store.connection(True) as db:
        db.execute(
            "INSERT INTO media(id,scope,node,original,path,mime,status) VALUES(?,?,?,?,?,?,?)",
            (
                "image-original",
                owner.scope,
                ids[1],
                original,
                str(path),
                "image/png",
                "ready",
            ),
        )
    owner.context = SimpleNamespace(conversation_manager=MemoryConversationManager())
    bridge = HistoryBridge(owner)
    try:
        projected = owner.messages(owner.store.view(owner.scope)["nodes"])
        assert projected[1]["content"][0]["image_url"]["url"].startswith("data:image/")
        # Native history can already contain the local representation when its
        # first mapping is built; the immutable archive still retains the URL.
        synchronized = owner.store.sync_history(
            owner.scope, projected, projected, candidates=bridge.candidates(owner.scope)
        )
        assert synchronized["ids"] == ids
        assert (
            owner.store.node(owner.scope, ids[1])["payload"]["message"] == messages[1]
        )
    finally:
        bridge.close()
