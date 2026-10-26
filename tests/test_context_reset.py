"""Reset native conversation pointers with retained archives and shared memory."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
import pytest_asyncio
from astrbot.builtin_stars.astrbot.group_chat_context import GroupChatContext
from astrbot.builtin_stars.astrbot.main import Main
from astrbot.core.conversation_mgr import ConversationManager
from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.utils.active_event_registry import active_event_registry
from astrbot.core.utils.shared_preferences import SharedPreferences
from astrbot.dashboard.api.plugins import _call_plugin_extension
from fastapi import FastAPI, Request

from astrbot_plugin_lcm.main import LCM, HistoryTool
from astrbot_plugin_lcm.tests.test_integration import event


class Active:
    def __init__(self, origin):
        self.unified_msg_origin = origin
        self.extras = {}
        self.stop_event = Mock()

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value


@pytest_asyncio.fixture
async def reset(tmp_path, monkeypatch):
    from astrbot.core import conversation_mgr

    db = SQLiteDatabase(str(tmp_path / "native.sqlite3"))
    await db.initialize()
    preferences = SharedPreferences(db)
    monkeypatch.setattr(conversation_mgr, "sp", preferences)
    manager = ConversationManager(db)
    origin = "test:GroupMessage:room"
    cid = await manager.new_conversation(origin, persona_id="selected-persona")
    old_history = [
        {"role": "user", "content": f"previous personality {i}"} for i in range(14)
    ]
    await manager.update_conversation(origin, cid, history=old_history)
    other_cid = await manager.new_conversation("other")
    group = GroupChatContext(None, None)
    group.raw_records[origin].append("old pending group message")
    group.raw_records["other"].append("other group message")
    instance = object.__new__(Main)
    instance.group_chat_context = group
    shared = {
        "persona": "current personality and memory",
        "notebook": ["remembered fact"],
    }
    registered = []
    context = SimpleNamespace(
        conversation_manager=manager,
        get_all_stars=lambda: [
            SimpleNamespace(star_cls=instance),
            SimpleNamespace(star_cls=shared),
        ],
        get_config=lambda **_: {"timezone": "Asia/Shanghai"},
        register_web_api=lambda *args: registered.append(args),
        add_llm_tools=lambda *args: None,
        registered_web_apis=registered,
    )
    with patch("astrbot_plugin_lcm.main.StarTools.get_data_dir", return_value=tmp_path):
        owner = LCM(context, {"base_tokens": 20})
    scope = owner.store.scope(origin, cid, "Test room")
    owner.store.import_history(scope, old_history)
    job_id = owner.store.plan_job(scope, 20)
    assert job_id
    job = owner.store.next_job()
    owner.store.job_result(job_id, "old")
    owner.store.publish(job)
    assert any(n["level"] > 0 for n in owner.store.view(scope)["nodes"])
    app = FastAPI()
    app.state.core_lifecycle = SimpleNamespace(star_context=context)

    @app.api_route("/api/{name}", methods=["GET", "POST"])
    async def api(name: str, request: Request):
        return await _call_plugin_extension(
            "astrbot_plugin_lcm/" + name, request, "isolated"
        )

    try:
        yield SimpleNamespace(
            owner=owner,
            manager=manager,
            origin=origin,
            cid=cid,
            scope=scope,
            old_history=old_history,
            app=app,
            group=group,
            shared=shared,
            other_cid=other_cid,
        )
    finally:
        for task in owner.media_tasks:
            task.cancel()
        await asyncio.gather(*owner.media_tasks, return_exceptions=True)
        await preferences.close()
        await db.engine.dispose()


@pytest.mark.asyncio
async def test_new_context_has_no_old_history_or_summaries_and_keeps_persona_memory(
    reset,
):
    s = reset
    before = copy.deepcopy(s.owner.store.history(s.scope))
    old_view = copy.deepcopy(s.owner.store.view(s.scope))
    shared = copy.deepcopy(s.shared)
    result = await s.owner.reset_context(s.scope)
    assert result["scope"] != s.scope and result["cid"] != s.cid
    assert await s.manager.get_curr_conversation_id(s.origin) == result["cid"]
    fresh = await s.manager.get_conversation(s.origin, result["cid"])
    assert json.loads(fresh.history or "[]") == []
    assert fresh.persona_id == "selected-persona"
    assert s.owner.store.view(result["scope"])["nodes"] == []
    assert s.owner.store.history(s.scope) == before
    assert s.owner.store.view(s.scope) == old_view
    assert (
        json.loads((await s.manager.get_conversation(s.origin, s.cid)).history)
        == s.old_history
    )
    assert s.shared == shared
    assert s.origin not in s.group.raw_records
    assert list(s.group.raw_records["other"]) == ["other group message"]
    assert await s.manager.get_curr_conversation_id("other") == s.other_cid
    # The next real archival/preparation path starts from the new conversation.
    ev = event()
    ev.unified_msg_origin = s.origin
    ev.get_sender_id = lambda: "alice"
    ev.get_self_id = lambda: "bot"
    ev.get_messages = lambda: []
    ev.get_sender_name = lambda: "Alice"
    ev.message_str = "first new message"
    ev.message_obj = SimpleNamespace(timestamp=1, message_id="first-new")
    await s.owner.archive(ev)
    req = ProviderRequest(
        prompt=ev.message_str,
        contexts=[],
        system_prompt=s.shared["persona"],
        conversation=fresh,
    )
    await s.owner.prepare(ev, req)
    assert req.contexts == []
    assert req.system_prompt == shared["persona"]
    assert ev.get_extra("lcm_prepared")["scope"] == result["scope"]
    search = HistoryTool(s.owner, result["scope"], "lcm_search", "", {}, [])
    assert json.loads(await search.call(None, query="previous"))["records"] == []
    assert s.owner.store.search(s.scope, "previous")["records"]
    ev.cleanup_temporary_local_files()


@pytest.mark.asyncio
async def test_reset_stops_only_target_session_and_stale_retry_cannot_reset_again(
    reset,
):
    current, other = Active(reset.origin), Active("other")
    callback = Mock()
    active_event_registry.register(current)
    active_event_registry.register(other)
    active_event_registry.register_agent_stop_callback(current, callback)
    try:
        result = await reset.owner.reset_context(reset.scope)
        current.stop_event.assert_called_once()
        callback.assert_called_once()
        other.stop_event.assert_not_called()
        assert result["stopped"] == 1
        with pytest.raises(ValueError, match="当前对话已变化"):
            await reset.owner.reset_context(reset.scope)
        assert (
            await reset.manager.get_curr_conversation_id(reset.origin) == result["cid"]
        )
    finally:
        active_event_registry.unregister(current)
        active_event_registry.unregister(other)


@pytest.mark.asyncio
async def test_post_route_and_current_session_labels_keep_archive_browsable(reset):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=reset.app), base_url="http://isolated"
    ) as client:
        sessions = (await client.get("/api/sessions")).json()["sessions"]
        assert next(s for s in sessions if s["id"] == reset.scope)["current"]
        malformed = await client.post(
            "/api/reset", json={"scope": reset.scope, "delete": True}
        )
        assert malformed.status_code == 400
        unchanged = await reset.manager.get_curr_conversation_id(reset.origin)
        assert unchanged == reset.cid
        assert (await client.get("/api/reset", params={"scope": reset.scope})).json()[
            "status"
        ] == "error"
        response = await client.post("/api/reset", json={"scope": reset.scope})
        assert response.status_code == 200
        result = response.json()
        sessions = (await client.get("/api/sessions")).json()["sessions"]
        assert not next(s for s in sessions if s["id"] == reset.scope)["current"]
        assert next(s for s in sessions if s["id"] == result["scope"])["current"]
        assert (await client.get("/api/history", params={"scope": reset.scope})).json()[
            "records"
        ]
        assert (
            await client.post("/api/reset", json={"scope": reset.scope})
        ).status_code == 400
