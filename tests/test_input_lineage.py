"""Keep current input versions and their original records connected end to end."""

import base64
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from openai import AsyncOpenAI

from astrbot_plugin_lcm.bridge import RunnerBridge
from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.main import HistoryTool
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_effective_pipeline import MemoryConversationManager
from astrbot_plugin_lcm.tests.test_image_context import IMAGE, message
from astrbot_plugin_lcm.tests.test_integration import event, runner

owner = test_integration.owner


def revise(store, scope, before, after):
    store.sync_history(scope, [before], [before])
    store.sync_history(scope, [before], [after])
    return store.view(scope)["nodes"][0]["id"]


def test_reserve_current_version_preserves_original_and_rejects_other_owner(owner):
    original = {"role": "user", "content": "original"}
    raw = owner.store.append(owner.scope, original)
    current = {"role": "user", "content": "processed"}
    first = revise(owner.store, owner.scope, original, current)
    newest = {"role": "user", "content": "processed twice"}
    owner.store.sync_history(owner.scope, [current], [newest])
    last = owner.store.view(owner.scope)["nodes"][0]["id"]
    assert last != first != raw
    assert owner.store.reserve(owner.scope, [raw, first, last], "owner") == [last]
    with pytest.raises(ValueError, match="reserved"):
        owner.store.reserve(owner.scope, [raw], "other-owner")
    assert owner.store.node(owner.scope, last)["reservation"] == "owner"
    restored = owner.store.expand(owner.scope, last)["units"][0]["records"]
    assert restored[0]["id"] == raw
    assert restored[0]["payload"]["message"] == original
    assert not owner.store.node(owner.scope, raw)["active"]
    owner.store.release("owner")
    assert owner.store.node(owner.scope, last)["reservation"] is None


@pytest.mark.parametrize("invalid", ["deleted", "other_scope", "compacted", "cycle"])
def test_invalid_input_does_not_resurrect_or_partially_reserve(owner, invalid):
    first = owner.store.append(owner.scope, {"role": "user", "content": "valid"})
    bad = owner.store.append(owner.scope, {"role": "user", "content": "invalid"})
    if invalid == "deleted":
        values = [
            node["payload"]["message"]
            for node in owner.store.view(owner.scope)["nodes"]
        ]
        owner.store.sync_history(owner.scope, values, values)
        owner.store.sync_history(owner.scope, values, values[:1])
    elif invalid == "other_scope":
        scope = owner.store.scope("other", "other-conv")
        bad = owner.store.append(scope, {"role": "user", "content": "other"})
    else:
        with owner.store.connection(True) as db:
            db.execute("UPDATE nodes SET active=0 WHERE id=?", (bad,))
            if invalid == "cycle":
                db.execute(
                    "INSERT INTO retirements VALUES(?,?,?,'test-cycle',0)",
                    (owner.scope, bad, bad),
                )
    with pytest.raises(ValueError, match="compacted_or_reserved"):
        owner.store.reserve(owner.scope, [first, bad], "owner")
    assert owner.store.node(owner.scope, first)["reservation"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("image_status", ["ready", "missing"])
async def test_admission_revision_reaches_native_request_and_keeps_image_lineage(
    owner, image_status
):
    expired = "https://expired.invalid/old-qq-image.png"
    old = message("old history", reference=expired)
    old_id = owner.store.append(owner.scope, old)
    owner.store.sync_history(owner.scope, [old], [old])
    current = {"role": "user", "content": "current original"}
    raw = owner.store.append(owner.scope, current)
    processed = copy.deepcopy(current)
    processed["content"] = "current processed"
    old_processed = copy.deepcopy(old)
    old_processed["content"][0]["text"] = "old processed history"
    owner.store.sync_history(
        owner.scope, [old], [old, current], candidates=[{"id": raw, "message": current}]
    )
    owner.store.sync_history(owner.scope, [old, current], [old_processed, processed])
    versions = owner.store.view(owner.scope)["nodes"]
    old_version, current_version = [node["id"] for node in versions]
    assert old_id in versions[0]["children"] and raw in versions[1]["children"]
    path = owner.media.root / "cached.png"
    path.write_bytes(base64.b64decode(IMAGE.split(",", 1)[1]))
    with owner.store.connection(True) as db:
        db.execute(
            "INSERT INTO media(id,scope,node,original,status,path,mime) VALUES(?,?,?,?,?,?,?)",
            (
                "original-image",
                owner.scope,
                old_id,
                expired,
                image_status,
                str(path),
                "image/png",
            ),
        )
    manager = MemoryConversationManager([old_processed, processed])
    owner.context = SimpleNamespace(conversation_manager=manager)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    bridge = RunnerBridge(owner)
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
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert expired not in json.dumps(body)
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
                        "message": {"role": "assistant", "content": "reply ok"},
                    }
                ],
            },
        )

    provider.client = AsyncOpenAI(
        api_key="offline-marker",
        base_url="https://offline.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    ev = event()
    ev.set_extra("lcm_input", {"scope": owner.scope, "id": raw})
    req = ProviderRequest(
        prompt="current processed",
        contexts=manager.history,
        conversation=SimpleNamespace(cid="conv", token_usage=0),
    )
    try:
        await owner.prepare(ev, req)
        state = ev.get_extra("lcm_prepared")
        assert state["inputs"] == [current_version]
        assert [node["id"] for node in state["base"]["nodes"]] == [old_version]
        loop = await runner(owner, ev, req, provider)
        while not loop.done():
            async for _ in loop.step():
                pass
        assert loop.get_final_llm_resp().completion_text == "reply ok"
        assert len(bodies) == 1
        body = json.dumps(bodies[0], ensure_ascii=False)
        assert body.count("current processed") == 1 and "old processed history" in body
        assert ("data:image/png;base64" in body) == (image_status == "ready")
        assert (
            owner.media.records(owner.scope, old_version)[0]["id"] == "original-image"
        )
        restored = owner.store.expand(owner.scope, current_version)["units"][0][
            "records"
        ]
        assert restored[0]["id"] == raw and restored[0]["payload"]["message"] == current
        tool = HistoryTool(owner, owner.scope, "lcm_expand", "expand", {}, [])
        retrieved = json.loads(await tool.call(None, id=current_version))
        assert retrieved["units"][0]["records"][0]["id"] == raw
        original_image = json.loads(await tool.call(None, id=old_version))
        assert (
            original_image["units"][0]["records"][0]["media"][0]["id"]
            == "original-image"
        )
        if image_status == "ready":
            reopened = await tool.call(None, id="original-image")
            assert base64.b64decode(reopened.content[1].data) == path.read_bytes()
    finally:
        ev.cleanup_temporary_local_files()
        bridge.close()
        history.close()
        await provider.client.close()


@pytest.mark.asyncio
async def test_refresh_follows_another_revision_before_runner_start(owner):
    original = {"role": "user", "content": "current original"}
    raw = owner.store.append(owner.scope, original)
    manager = MemoryConversationManager([original])
    owner.context = SimpleNamespace(conversation_manager=manager)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    ev = event()
    ev.set_extra("lcm_input", {"scope": owner.scope, "id": raw})
    req = ProviderRequest(
        prompt="current original",
        conversation=SimpleNamespace(cid="conv", token_usage=0),
    )
    try:
        await owner.prepare(ev, req)
        state = ev.get_extra("lcm_prepared")
        assert state["inputs"] == [raw]
        await manager.update_conversation(
            "group", "conv", history=[{"role": "user", "content": "processed"}]
        )
        newest = owner.store.view(owner.scope)["nodes"][0]["id"]
        await history.refresh_request(ev, req)
        assert state["inputs"] == [newest] and newest != raw
        assert req.contexts == []
        assert (
            owner.store.node(owner.scope, newest)["reservation"] == state["reservation"]
        )
    finally:
        ev.cleanup_temporary_local_files()
        history.close()
