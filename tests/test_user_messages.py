"""Keep human message metadata short without changing the actual content."""

import copy
import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from astrbot.core.conversation_mgr import ConversationManager
from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.provider.entities import ProviderRequest

from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.image_context import ImageContext
from astrbot_plugin_lcm.tests.test_integration import owner  # noqa: F401
from astrbot_plugin_lcm.user_message import event_message, node_message, user_message

ZONE = ZoneInfo("Asia/Shanghai")
STAMP = datetime(2026, 10, 26, 18, 4, 12, tzinfo=ZONE).timestamp()
HEADER = "发送人：小幸\n发送时间：2026-10-26 18:04:12\n"
REMINDER = (
    "<system_reminder>User ID: alice, Nickname: 小幸\n"
    "Group name: 测试群\n"
    "Current datetime: 2026-10-26T18:04:00+08:00, Timezone: Asia/Shanghai, "
    "Weekday: Monday</system_reminder>"
)


@pytest.mark.parametrize("structured", [False, True])
def test_old_duplicate_metadata_becomes_two_lines(structured):
    body = "[小幸(alice)/2026-10-26 18:04:12] 原话\n第二行"
    raw = {
        "role": "user",
        "content": (
            [{"type": "text", "text": body}, {"type": "text", "text": REMINDER}]
            if structured
            else body + "\n" + REMINDER
        ),
    }
    before = copy.deepcopy(raw)
    result = user_message(raw, ZONE)
    expected = HEADER + "原话\n第二行"
    assert result["content"] == (
        [{"type": "text", "text": expected}] if structured else expected
    )
    assert raw == before
    assert user_message(result, ZONE) == result


def test_archive_time_and_sender_override_request_generation_time():
    raw = {"role": "user", "content": "原话\n" + REMINDER}
    node = {
        "sender": "小幸(alice)",
        "timestamp": STAMP,
        "payload": {"extra": {}},
    }
    assert node_message(raw, node, ZONE)["content"] == HEADER + "原话"
    legacy = {**node, "payload": {"extra": {"time_unknown": True}}}
    assert "发送时间：2026-10-26 18:04:00" in node_message(raw, legacy, ZONE)["content"]


def test_media_and_quotes_keep_order_and_contents():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    quoted = {
        "type": "text",
        "text": "<Quoted Message>\n" + REMINDER + "\n</Quoted Message>",
    }
    raw = {
        "role": "user",
        "content": [
            {"type": "text", "text": "[2026-10-26 18:04:12 小幸(alice)] 看图"},
            image,
            quoted,
            {"type": "text", "text": REMINDER},
        ],
    }
    result = user_message(raw, ZONE)
    assert result["content"] == [
        {"type": "text", "text": HEADER + "看图"},
        image,
        quoted,
    ]


def test_current_event_and_repeated_formatting_preserve_parenthesized_nickname():
    event = SimpleNamespace(
        get_sender_name=lambda: "小幸(test)",
        get_sender_id=lambda: "alice",
        message_obj=SimpleNamespace(timestamp=STAMP),
    )
    raw = {"role": "user", "content": "本轮原话"}
    result = event_message(raw, event, ZONE)
    assert result["content"] == HEADER.replace("小幸", "小幸(test)") + "本轮原话"
    assert user_message(result, ZONE) == result
    event.get_sender_name = lambda: ""
    assert "发送人：alice\n" in event_message(raw, event, ZONE)["content"]


def test_unknown_history_and_tool_user_messages_are_not_fabricated():
    for raw in (
        {"role": "user", "content": "没有身份资料的旧正文"},
        {"role": "user", "content": "[历史记录：机器人已发送图片]"},
        {"role": "assistant", "content": "正常回复"},
        {"role": "tool", "content": REMINDER},
    ):
        assert node_message(raw, {"sender": "机器人", "timestamp": STAMP}, ZONE) == raw
    result = user_message({"role": "user", "content": "正文"}, ZONE, sender="小幸")
    assert result["content"] == "发送人：小幸\n发送时间：未知\n正文"


def test_archive_without_nickname_preserves_body_whitespace():
    raw = {"role": "user", "content": "[2026-10-26 18:04:12 (alice)] 原话\n\n"}
    result = user_message(raw, ZONE)
    assert result["content"] == HEADER.replace("小幸", "alice") + "原话\n\n"


def test_request_uses_event_time_and_only_one_identity_header(owner):
    owner.context = SimpleNamespace(get_config=lambda **kw: {"timezone": "Asia/Shanghai"})
    extras = {}
    event = SimpleNamespace(
        unified_msg_origin="group",
        get_sender_name=lambda: "小幸",
        get_sender_id=lambda: "alice",
        message_obj=SimpleNamespace(timestamp=STAMP),
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
    )
    req = ProviderRequest(prompt="当前原话", system_prompt="人设")
    req.extra_user_content_parts = [{"type": "text", "text": REMINDER}]
    owner.context_rules().configure_request(event, req)
    assert req.prompt == HEADER + "当前原话"
    assert req.extra_user_content_parts == []
    owner.context_rules().configure_request(event, req)
    assert req.prompt == HEADER + "当前原话"


@pytest.mark.asyncio
async def test_history_view_save_and_image_projection_share_short_metadata(owner, tmp_path):
    db = SQLiteDatabase(str(tmp_path / "native.db"))
    manager = ConversationManager(db)
    cid = await manager.new_conversation("group")
    owner.context = SimpleNamespace(
        conversation_manager=manager,
        get_config=lambda **kw: {"timezone": "Asia/Shanghai"},
    )
    scope = owner.store.scope("group", cid)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    raw = {
        "role": "user",
        "content": [{"type": "text", "text": "看图\n" + REMINDER}, image],
    }
    ident = owner.store.append(scope, raw, sender="小幸(alice)", timestamp=STAMP)
    bridge = HistoryBridge(owner)
    try:
        await manager.update_conversation("group", cid, history=[raw])
        actual = json.loads((await manager.get_conversation("group", cid)).history)
        assert actual[0]["content"] == [{"type": "text", "text": HEADER + "看图"}, image]
        nodes = owner.view(scope)["nodes"]
        assert owner.messages(nodes) == actual
        projected = await ImageContext(owner).history(actual, 1, nodes)
        assert projected == actual
        assert "历史消息 id=" not in str(projected)
        assert owner.store.node(scope, ident)["payload"]["message"] == raw
    finally:
        bridge.close()
        await db.engine.dispose()
