"""Verify model times, official history writes and decision-only metadata."""

import copy
import json
import re
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from astrbot.core.conversation_mgr import ConversationManager
from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.provider.entities import ProviderRequest

from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.main import HistoryTool, timestamp
from astrbot_plugin_lcm.model_time import model_times
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_reply_gate.decision_input import build_input

owner = test_integration.owner


def test_zone_conversion_keeps_local_clock_without_zone_names():
    raw = {
        "timestamp": 1792869080,
        "content": "<system_reminder>Current datetime: 2026-10-24T19:11:20+00:00, Timezone: Asia/Shanghai, Weekday: 周日</system_reminder>",
        "sent_at": "2026-10-25T03:11:20+08:00",
    }
    result = model_times(raw, ZoneInfo("Asia/Shanghai"))
    assert "2026-10-25 03:11:20" in result["content"]
    assert result["sent_at"] == "2026-10-25 03:11:20"
    assert "+08:00" not in str(result) and "Asia/Shanghai" not in str(result)
    assert "Timezone" not in str(result)
    assert "+00:00" in raw["content"]
    assert timestamp("2026-10-25 03:11:20", ZoneInfo("Asia/Shanghai")) == timestamp(
        "2026-10-24T19:11:20Z"
    )


@pytest.mark.asyncio
async def test_timestamp_saved_in_native_sqlite_and_reused_for_model_history(
    owner, tmp_path
):
    db = SQLiteDatabase(str(tmp_path / "native.db"))
    manager = ConversationManager(db)
    cid = await manager.new_conversation("group")
    owner.context = SimpleNamespace(
        conversation_manager=manager,
        get_config=lambda **kw: {"timezone": "Asia/Shanghai"},
    )
    scope = owner.store.scope("group", cid)
    raw = {"role": "assistant", "content": "实际机器人输出"}
    ident = owner.store.append(scope, raw, timestamp=1792869080)
    bridge = HistoryBridge(owner)
    try:
        await manager.update_conversation("group", cid, history=[raw])
        actual = json.loads((await manager.get_conversation("group", cid)).history)
        assert re.match(
            r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} 机器人\] 实际机器人输出$",
            actual[0]["content"],
        )
        assert owner.messages(owner.view(scope)["nodes"]) == actual
        assert owner.store.node(scope, ident)["payload"]["message"] == raw
        # Re-saving and appending a parallel tail must not duplicate the stamp.
        tail = {"role": "assistant", "content": "并行输出"}
        owner.store.append(scope, tail, timestamp=1792869081)
        await manager.update_conversation("group", cid, history=[*actual, tail])
        saved = json.loads((await manager.get_conversation("group", cid)).history)
        assert len(saved) == 2 and saved[0] == actual[0]
        assert saved[1]["content"].count("机器人]") == 1
        assert owner.messages(owner.view(scope)["nodes"]) == saved
    finally:
        bridge.close()
        await db.engine.dispose()


@pytest.mark.asyncio
async def test_minimal_runtime_and_judge_metadata_without_tool_guidance(owner):
    owner.context = SimpleNamespace(
        get_config=lambda **kw: {"timezone": "Asia/Shanghai"}
    )
    extra = {}
    event = SimpleNamespace(
        unified_msg_origin="group",
        get_sender_id=lambda: "alice",
        get_sender_name=lambda: "昵称",
        get_extra=lambda k, d=None: extra.get(k, d),
        set_extra=lambda k, v: extra.__setitem__(k, v),
    )
    request = ProviderRequest(
        prompt="当前聊天",
        system_prompt="人设",
        contexts=[{"role": "assistant", "content": "历史回答"}],
    )
    request.extra_user_content_parts = [
        {
            "type": "text",
            "text": "<system_reminder>User ID: alice, Nickname: 昵称\nCurrent datetime: 2026-10-25T03:11+08:00, Timezone: Asia/Shanghai</system_reminder>",
        },
        {"type": "text", "text": "工具指导：不要进入判断"},
    ]
    owner.context_rules().configure_request(event, request)
    temporary = owner.context_rules().temporary(event)
    assert (
        "当前时间：" in temporary
        and "User ID" not in temporary
        and "Nickname" not in temporary
    )
    assert "Asia/Shanghai" not in temporary and "+08:00" not in temporary
    assert "LCM 历史" not in request.system_prompt
    before = copy.deepcopy(request.extra_user_content_parts)
    history, current = await build_input(owner.context, event, request)
    assert "历史回答" in str(history) and "当前聊天" in str(current)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", current["current_time"])
    assert "工具指导" not in str(current) and request.extra_user_content_parts == before


@pytest.mark.asyncio
async def test_history_tool_time_and_unknown_legacy_time(owner):
    owner.context = SimpleNamespace(
        get_config=lambda **kw: {"timezone": "Asia/Shanghai"}
    )
    raw = {"role": "assistant", "content": "工具检索用的输出"}
    owner.store.append(owner.scope, raw, timestamp=1792869080)
    tool = HistoryTool(owner, owner.scope, "lcm_search", "", {}, [])
    result = json.loads(await tool.call(None, query="工具检索"))
    assert result["ok"] and isinstance(result["records"][0]["timestamp"], str)
    assert "+08:00" not in str(result) and "Asia/Shanghai" not in str(result)
    unknown = owner.store.append(owner.scope, raw, extra={"time_unknown": True})
    assert owner.messages([owner.store.node(owner.scope, unknown)])[0] == raw


@pytest.mark.asyncio
async def test_voice_exposes_tool_without_adding_system_style():
    from astrbot_plugin_voice_reply.main import VoiceReply

    voice = object.__new__(VoiceReply)
    voice.config = {"enabled": True}
    voice.service = SimpleNamespace(ready=True)
    request = ProviderRequest(system_prompt="用户配置的人设和回复偏好")
    await voice.prepare(None, request)
    assert request.system_prompt == "用户配置的人设和回复偏好"
    assert "silence" in VoiceReply.send.__doc__


def test_gallery_recommendation_caveat_is_beside_candidates():
    from astrbot_plugin_meme_library.main import MemeLibrary

    gallery = object.__new__(MemeLibrary)
    gallery.library = SimpleNamespace(
        latest_seq=lambda: 1,
        unseen=lambda origin: 0,
        recommend=lambda: [{"id": "candidate", "name": "图名", "description": "图意"}],
    )
    extra = {}
    event = SimpleNamespace(
        unified_msg_origin="isolated",
        get_extra=lambda key, default=None: extra.get(key, default),
        set_extra=lambda key, value: extra.__setitem__(key, value),
    )
    rules = gallery.lcm_context_rules()
    assert "system" not in rules
    temporary = rules["temporary"](event)
    assert "candidate" in temporary and "只是图库的一部分" in temporary
    assert "可以用也可以不用" in temporary and "meme_search" in temporary
