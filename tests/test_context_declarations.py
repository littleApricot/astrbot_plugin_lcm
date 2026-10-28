"""Exercise declared inputs through the native runner and immutable archive."""

import copy
import json
from types import SimpleNamespace

import pytest
from astrbot.core.agent.message import TextPart
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.astr_main_agent_resources import TOOL_CALL_PROMPT
from astrbot.core.provider.entities import LLMResponse

from astrbot_plugin_lcm.bridge import RunnerBridge
from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_effective_pipeline import (
    PRIVATE,
    ChooseMeme,
    MemoryConversationManager,
    native_save,
    run_to_end,
)
from astrbot_plugin_lcm.tests.test_integration import FakeProvider, prepare, runner
from astrbot_plugin_meme_library.main import MemeLibrary
from astrbot_plugin_meme_library.store import Library
from astrbot_plugin_reply_gate.decision_input import build_input
from astrbot_plugin_self_prompt.tests.support import Sandbox

owner = test_integration.owner
OLD_CLOCK = (
    "<system_reminder>Current datetime: 2026-10-24T01:00+08:00</system_reminder>"
)
LOCAL_CLOCK = "<system_reminder>Current datetime: 2026-10-24 01:00:00</system_reminder>"


def install_gallery(owner, tmp_path):
    gallery = object.__new__(MemeLibrary)
    gallery.config = {}
    gallery.library = Library(tmp_path / "gallery")
    star = SimpleNamespace(star_cls=gallery, activated=True)
    owner.context.get_all_stars = lambda: [star]
    owner.context.get_registered_star = lambda name: (
        star if name == "astrbot_plugin_meme_library" else None
    )
    owner.context.get_config = lambda **kw: {"timezone": "Asia/Shanghai"}
    return gallery


@pytest.mark.asyncio
async def test_native_memory_preparation_declares_latest_catalog_separately(owner):
    sandbox = await Sandbox().open()
    try:
        owner.context = sandbox.context
        memory_star = SimpleNamespace(star_cls=sandbox.plugin, activated=True)
        lcm_star = SimpleNamespace(star_cls=owner, activated=True)
        owner.context.get_all_stars = lambda: [memory_star, lcm_star]
        owner.context.get_registered_star = lambda name: (
            lcm_star if name == "astrbot_plugin_lcm" else memory_star
        )
        ev = sandbox.message(2)
        req = await sandbox.prepare(ev)
        rules = owner.context_rules()
        rules.configure_request(ev, req)
        assert "来源：astrbot_plugin_self_prompt" in req.system_prompt
        assert "# 你维护的固定设定" not in req.system_prompt
        assert req.extra_user_content_parts == []
        created = json.loads(
            await sandbox.plugin.notebook_create(ev, "流程", "办事步骤", "具体步骤")
        )
        assert created["ok"]
        temporary = rules.temporary(ev)
        assert "流程" not in temporary and "办事步骤" not in temporary
        assert "具体步骤" not in temporary
        assert "流程" not in req.system_prompt
        repeated = rules.system(ev, req.system_prompt)
        assert "流程" in repeated and "办事步骤" in repeated
        assert "具体步骤" not in repeated
        assert repeated.count("来源：astrbot_plugin_self_prompt") == 1
    finally:
        await sandbox.close()


@pytest.mark.asyncio
async def test_native_turn_keeps_private_tools_but_only_detaches_temporary_inputs(
    owner, tmp_path
):
    old = [{"role": "user", "content": OLD_CLOCK}]
    manager = MemoryConversationManager(old)
    owner.context = SimpleNamespace(conversation_manager=manager)
    gallery = install_gallery(owner, tmp_path)
    owner.store.append(owner.scope, old[0])
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    try:
        ev, req = await prepare(owner, "当前消息", base=old)
        ev.set_extra("meme_library_temporary", "本轮图库提示")
        req.extra_user_content_parts.append(
            TextPart(
                text="<system_reminder>User ID: alice, Nickname: 小幸\nCurrent datetime: 2026-10-25T02:00+08:00</system_reminder>"
            )
        )
        req.system_prompt += (
            TOOL_CALL_PROMPT
            + "\n## Skills\n不可用技能目录\nUser has not enabled the Computer Use feature. "
            + "You cannot use shell or Python to perform skills. "
            + "If you need to use these capabilities, ask the user to enable Computer Use in the AstrBot WebUI -> Config."
        )
        await owner.remove_duplicate_context(ev, req)
        lookup = ChooseMeme()
        lookup.name = "meme_search"
        req.func_tool.add_tool(lookup)
        provider = FakeProvider(
            [
                LLMResponse(
                    role="assistant",
                    tools_call_name=["meme_search"],
                    tools_call_args=[{}],
                    tools_call_ids=["private-search"],
                ),
                LLMResponse(role="assistant", completion_text="正常文字回复"),
            ]
        )
        loop = await runner(owner, ev, req, provider)
        loop.tool_executor = FunctionToolExecutor()
        await run_to_end(loop)
        sent = json.dumps(
            [m.model_dump() for m in provider.calls[-1]["contexts"]], ensure_ascii=False
        )
        assert PRIVATE in sent and "本轮图库提示" in sent
        assert "来源：astrbot_plugin_meme_library" in sent
        assert "never return an empty response" not in sent
        assert "不可用技能目录" not in sent
        assert LOCAL_CLOCK in sent
        assert "Asia/Shanghai" not in sent and "+08:00" not in sent
        assert "User ID:" not in sent
        await native_save(manager, ev, req, loop)
        native = json.dumps(manager.history, ensure_ascii=False)
        assert PRIVATE in native  # Original execution is retained.
        assert "本轮图库提示" not in native and "时区：Asia/Shanghai" not in native
        assert "2026-10-25T02:00+08:00" not in native
        assert LOCAL_CLOCK in native
        _, next_req = await prepare(owner, "下一条")
        effective = json.dumps(next_req.contexts, ensure_ascii=False)
        assert PRIVATE not in effective and "正常文字回复" in effective
        assert LOCAL_CLOCK in effective
        judged, _ = await build_input(owner.context, ev, next_req)
        assert PRIVATE not in str(judged) and LOCAL_CLOCK in str(judged)
        snapshot = owner.store.requests(owner.scope, ev.get_extra("lcm_run"))[0]
        captured = owner.store.request(owner.scope, snapshot["id"])["payload"]
        assert captured["context_sources"]["temporary"]
    finally:
        execution.close()
        history.close()
        gallery.library.db.close()


def test_compaction_uses_effective_text_and_retires_hidden_sources(owner, tmp_path):
    gallery = install_gallery(owner, tmp_path)
    try:
        gallery.library.cache_sent_preview("picture", ["data:image/png;base64,AA=="])
        records = [
            {"role": "user", "content": OLD_CLOCK + "有效旧消息" * 100},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "s",
                        "type": "function",
                        "function": {"name": "meme_search", "arguments": "{}"},
                    },
                    {
                        "id": "e",
                        "type": "function",
                        "function": {"name": "meme_edit", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "s", "content": PRIVATE},
            {"role": "tool", "tool_call_id": "e", "content": "已修改描述"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "send",
                        "type": "function",
                        "function": {"name": "meme_send", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "send",
                "content": json.dumps(
                    {
                        "ok": True,
                        "sent": True,
                        "id": "picture",
                        "name": "图名",
                        "description": "图意",
                        "sent_at": "2026-10-24T01:01+08:00",
                    }
                ),
            },
        ]
        originals = []
        run = owner.store.begin(owner.scope, "event", owner.store.view(owner.scope), [])
        for message in records:
            originals.append(owner.store.append(owner.scope, message, round_id=run))
        owner.store.finish(run)
        for _ in range(4):
            owner.store.append(
                owner.scope, {"role": "user", "content": "保留的新资料" * 1000}
            )
        effective = owner.messages(owner.view(owner.scope)["nodes"])
        assert PRIVATE not in str(effective)
        assert "已修改描述" in str(effective) and "机器人已发送图片" in str(effective)
        projected = {n["id"]: n for n in owner.view(owner.scope)["nodes"]}
        assert not owner.store.search(owner.scope, PRIVATE, projected=projected)[
            "records"
        ]
        raw_before = [
            copy.deepcopy(owner.store.node(owner.scope, ident)["payload"])
            for ident in originals
        ]
        ident = owner.store.plan_job(
            owner.scope, 1000, nodes=owner.compression_nodes(owner.scope)
        )
        job = owner.store.next_job()
        assert ident and job["id"] == ident
        assert set(originals).issubset(set(job["sources"]))
        material = owner.messages(
            [owner.store.node(owner.scope, i) for i in job["sources"]]
        )
        assert PRIVATE not in str(material) and LOCAL_CLOCK in str(material)
        owner.store.job_result(ident, "旧消息、描述修改和发图事实。")
        assert owner.store.publish(job)
        assert PRIVATE not in str(owner.messages(owner.view(owner.scope)["nodes"]))
        assert [
            owner.store.node(owner.scope, i)["payload"] for i in originals
        ] == raw_before
    finally:
        gallery.library.db.close()
