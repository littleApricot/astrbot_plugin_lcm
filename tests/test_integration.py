"""Run the real AstrBot tool loop with deterministic providers, never live keys."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import Message, TextPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import ToolSet
from astrbot.core.agent.tool_executor import BaseFunctionToolExecutor
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from astrbot_plugin_lcm.bridge import RunnerBridge
from astrbot_plugin_lcm.main import LCM
from astrbot_plugin_lcm.media import MediaArchive
from astrbot_plugin_lcm.store import Store


class FakeProvider:
    def __init__(self, responses=None, pause=None):
        self.provider_config = {"id": "isolated", "max_context_tokens": 1000000}
        self.responses = responses or [
            LLMResponse(role="assistant", completion_text="最终回复")
        ]
        self.calls = []
        self.pause = pause

    def get_model(self):
        return "deterministic-test-provider"

    async def text_chat(self, **kw):
        self.calls.append(
            copy.deepcopy(
                {k: v for k, v in kw.items() if k not in ("abort_signal", "func_tool")}
            )
        )
        if self.pause:
            await self.pause.wait()
        return self.responses.pop(0)


class DummyExecutor(BaseFunctionToolExecutor):
    async def execute(self, *args, **kwargs):
        yield None


def event(mid="new"):
    extras = {}
    return SimpleNamespace(
        unified_msg_origin="group",
        set_extra=lambda k, v: extras.__setitem__(k, v),
        get_extra=lambda k, d=None: extras.get(k, d),
        cleanup_temporary_local_files=lambda: None,
        get_result=lambda: None,
        trace=SimpleNamespace(record=lambda *a, **k: None),
    )


@pytest.fixture
def owner(tmp_path):
    obj = object.__new__(LCM)
    obj.config = {"base_tokens": 40}
    obj.store = Store(tmp_path / "history.db")
    obj.media = MediaArchive(obj.store)
    obj.wake = asyncio.Event()
    obj.scope_locks = {}
    obj.media_tasks = set()
    obj.context = SimpleNamespace()
    obj.scope = obj.store.scope("group", "conv")
    obj.store.import_history(obj.scope, [])
    return obj


@pytest.mark.asyncio
async def test_incoming_unanswered_events_are_archived_once_with_timezone(owner):
    from astrbot.api.message_components import Plain
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata

    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.self_id = "bot"
    message.session_id = message.group_id = "group"
    message.message_id = "stable-platform-id"
    message.sender = MessageMember(user_id="alice", nickname="小幸")
    message.message = [Plain("不用回复的群消息")]
    message.message_str = "不用回复的群消息"
    message.timestamp = 1792663200  # 2026-10-22 18:00:00 +08:00
    ev = AstrMessageEvent(
        message.message_str,
        message,
        PlatformMetadata("test", "isolated", "test"),
        "group",
    )
    owner.context = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conv"),
            get_conversation=AsyncMock(
                return_value=SimpleNamespace(
                    history='[{"role":"user","content":"旧历史"}]'
                )
            ),
        ),
        get_config=lambda **k: {"timezone": "Asia/Shanghai"},
    )
    await owner.archive(ev)
    await owner.archive(ev)
    state = ev.get_extra("lcm_input")
    records = owner.store.history(state["scope"])["records"]
    assert len(records) == 2
    incoming = next(n for n in records if n["id"] == state["id"])
    text = incoming["payload"]["message"]["content"][0]["text"]
    assert "2026-10-22T18:00:00+08:00" in text and "小幸(alice)" in text
    assert incoming["round_id"] is None
    assert len(owner.store.runs(state["scope"])) == 0


def test_invalid_provider_window_reserve_is_explicit(owner):
    with pytest.raises(ValueError, match="reserve"):
        owner.hard_limit(
            SimpleNamespace(
                provider_config={
                    "max_context_tokens": 80,
                    "custom_extra_body": {"max_tokens": 100},
                }
            )
        )


async def prepare(owner, text="current", base=None):
    ev = event(text)
    mid = owner.store.append(
        owner.scope, {"role": "user", "content": text}, source=text
    )
    ev.set_extra("lcm_input", {"scope": owner.scope, "id": mid})
    req = ProviderRequest(
        prompt=text,
        contexts=base or [],
        system_prompt="当前人格和长期记忆",
        conversation=SimpleNamespace(cid="conv", token_usage=0),
        func_tool=ToolSet(owner.history_tools()),
    )
    await owner.prepare(ev, req)
    req.extra_user_content_parts.append(
        TextPart(text="--- BEGIN CONTEXT---\nduplicate\n--- END CONTEXT ---")
    )
    req.extra_user_content_parts.append(TextPart(text="独立记忆提示"))
    await owner.remove_duplicate_context(ev, req)
    return ev, req


async def runner(owner, ev, req, provider, hooks=None):
    loop = ToolLoopAgentRunner()
    await loop.reset(
        provider=provider,
        request=req,
        run_context=ContextWrapper(context=SimpleNamespace(event=ev)),
        tool_executor=DummyExecutor(),
        agent_hooks=hooks or BaseAgentRunHooks(),
        enforce_max_turns=1,
    )
    return loop


@pytest.mark.asyncio
async def test_actual_runner_keeps_full_base_and_captures_provider_snapshot(owner):
    for i in range(6):
        owner.store.append(owner.scope, {"role": "user", "content": "旧消息" + str(i)})
    ev, req = await prepare(owner)
    req.extra_user_content_parts.append(TextPart(text="这一轮的新信息"))
    bridge = RunnerBridge(owner)
    try:
        provider = FakeProvider()
        loop = await runner(owner, ev, req, provider)
        async for _ in loop.step():
            pass
        assert (
            len(provider.calls[0]["contexts"]) == 9
        )  # system + six old + new + temporary
        assert "旧消息0" in json.dumps(
            [m.model_dump() for m in provider.calls[0]["contexts"]], ensure_ascii=False
        )
        run = ev.get_extra("lcm_run")
        assert owner.store.runs(owner.scope)[0]["status"] == "complete"
        snap = owner.store.request(
            owner.scope, owner.store.requests(owner.scope, run)[0]["id"]
        )
        text = json.dumps(snap, ensure_ascii=False)
        assert "当前人格和长期记忆" in text and "独立记忆提示" in text
        assert "duplicate" not in text
        assert [t["name"] for t in snap["payload"]["tools"]] == [
            "lcm_describe",
            "lcm_expand",
            "lcm_search",
        ]
        ev.cleanup_temporary_local_files()
        assert owner.store.runs(owner.scope)[0]["status"] == "complete"
    finally:
        bridge.close()


@pytest.mark.asyncio
async def test_two_parallel_rounds_keep_own_snapshots_and_outputs(owner):
    bridge = RunnerBridge(owner)
    try:
        a, ra = await prepare(owner, "A")
        b, rb = await prepare(owner, "B")
        wait = asyncio.Event()
        pa = FakeProvider(pause=wait)
        pb = FakeProvider()
        la = await runner(owner, a, ra, pa)
        lb = await runner(owner, b, rb, pb)

        async def run(loop):
            async for _ in loop.step():
                pass

        task = asyncio.create_task(run(la))
        await asyncio.sleep(0.02)
        await run(lb)
        assert not task.done()  # B did not wait for A's generation.
        wait.set()
        await task
        runs = owner.store.runs(owner.scope)
        assert len(runs) == 2 and all(r["status"] == "complete" for r in runs)
        for r in runs:
            expanded = owner.store.expand(owner.scope, r["id"])["units"][0]["records"]
            assert [n["payload"]["message"]["role"] for n in expanded] == [
                "user",
                "assistant",
            ]
        snap_a = owner.store.request(
            owner.scope,
            owner.store.requests(owner.scope, a.get_extra("lcm_run"))[0]["id"],
        )
        snap_b = owner.store.request(
            owner.scope,
            owner.store.requests(owner.scope, b.get_extra("lcm_run"))[0]["id"],
        )
        assert snap_a["run"] != snap_b["run"]
    finally:
        bridge.close()


@pytest.mark.asyncio
async def test_tool_loop_preserves_full_results_and_fixed_base_after_publish(owner):
    for i in range(4):
        owner.store.append(
            owner.scope, {"role": "user", "content": "旧消息" * 50}, tokens=20
        )
    ev, req = await prepare(owner)
    bridge = RunnerBridge(owner)
    try:
        provider = FakeProvider()
        loop = await runner(owner, ev, req, provider)
        original = [m.model_dump() for m in loop.run_context.messages]
        job_id = owner.store.plan_job(owner.scope, 40)
        job = owner.store.next_job()
        assert job["id"] == job_id
        owner.store.job_result(job_id, "较老内容摘要")
        owner.store.publish(job)
        assert [m.model_dump() for m in loop.run_context.messages] == original
        # Execute a second provider step over a full native tool-call/result pair.
        long_result = "FULL_TOOL_RESULT" * 5000
        loop.run_context.messages.extend(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "x", "arguments": "{}"},
                        }
                    ],
                ),
                Message(role="tool", content=long_result, tool_call_id="call-1"),
            ]
        )
        async for _ in loop.step():
            pass
        run = ev.get_extra("lcm_run")
        records = owner.store.expand(owner.scope, run)["units"][0]["records"]
        result = next(
            n for n in records if n["payload"]["message"].get("role") == "tool"
        )
        assert result["payload"]["message"]["content"] == long_result
        assert result["payload"]["message"]["tool_call_id"] == "call-1"
        assert loop.tool_result_overflow_dir is None
        latest = owner.messages(owner.store.view(owner.scope)["nodes"])
        assert any("较老内容摘要" in str(m) for m in latest)
        assert "旧消息" in str(provider.calls[0]["contexts"])
    finally:
        bridge.close()


@pytest.mark.asyncio
async def test_summary_worker_rejects_truncated_output_and_calls_no_event_hooks(owner):
    for i in range(4):
        owner.store.append(
            owner.scope, {"role": "user", "content": "历史" * 100}, tokens=30
        )
    response = LLMResponse(
        role="assistant", completion_text="摘要", raw_completion=None
    )
    response.raw_completion = SimpleNamespace(
        model_dump=lambda **k: {"choices": [{"finish_reason": "length"}]}
    )
    provider = FakeProvider([response])
    owner.context = SimpleNamespace(
        get_using_provider_async=lambda origin: asyncio.sleep(0, result=provider)
    )
    owner.logger = SimpleNamespace(warning=lambda *a: None)
    task = asyncio.create_task(owner.work())
    try:
        for _ in range(30):
            await asyncio.sleep(0.01)
            if (
                owner.store.view(owner.scope)["jobs"]
                and owner.store.view(owner.scope)["jobs"][0]["status"] == "retry"
            ):
                break
        jobs = owner.store.view(owner.scope)["jobs"]
        assert jobs[0]["status"] == "retry"
        assert "truncated" in jobs[0]["error"]
        assert all(n["level"] == 0 for n in owner.store.view(owner.scope)["nodes"])
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        {"choices": [{"finish_reason": "stop"}]},
        {"candidates": [{"finishReason": "STOP"}]},
    ],
)
async def test_summary_worker_publishes_complete_provider_result(owner, raw):
    for _ in range(4):
        owner.store.append(
            owner.scope, {"role": "user", "content": "历史资料" * 50}, tokens=20
        )
    response = LLMResponse(role="assistant", completion_text="中性历史摘要")
    response.raw_completion = raw
    provider = FakeProvider([response])
    owner.context = SimpleNamespace(
        get_using_provider_async=lambda origin: asyncio.sleep(0, result=provider)
    )
    owner.logger = SimpleNamespace(warning=lambda *a: None)
    task = asyncio.create_task(owner.work())
    try:
        for _ in range(50):
            await asyncio.sleep(0.01)
            if any(n["level"] for n in owner.store.view(owner.scope)["nodes"]):
                break
        view = owner.store.view(owner.scope)
        assert any(
            n["level"] == 1 and n["payload"]["text"] == "中性历史摘要"
            for n in view["nodes"]
        )
        assert len(provider.calls) == 1
        assert "来源节点" not in provider.calls[0]["prompt"]
        assert "必要时间" in provider.calls[0]["prompt"]
        assert len(owner.store.history(owner.scope)["records"]) == 4
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_actual_search_tool_round_and_existing_memory_collection(owner, tmp_path):
    from unittest.mock import AsyncMock

    from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor

    from astrbot_plugin_self_prompt.main import SelfPrompt
    from astrbot_plugin_self_prompt.memory_queue import MemoryQueue

    old = owner.store.append(owner.scope, {"role": "user", "content": "Alice 喜欢蓝猫"})
    ev, req = await prepare(owner, "你还记得 Alice 喜欢什么吗")
    ev.message_obj = SimpleNamespace(message_id="memory-current")
    ev.get_sender_id = lambda: "u1"
    ev.get_self_id = lambda: "bot"
    ev.get_sender_name = lambda: "Alice"
    ev.get_message_outline = lambda: "你还记得 Alice 喜欢什么吗"
    memory = object.__new__(SelfPrompt)
    memory.config = {}
    memory.memory_queue = MemoryQueue(tmp_path / "memory.sqlite3")
    memory.store = memory.memory_queue
    memory._selected_persona = AsyncMock(return_value="persona")
    memory._start_worker = lambda: None
    await memory.collect_history(ev)
    assert memory.memory_queue.pending_count() == 1

    class Hooks(BaseAgentRunHooks):
        async def on_agent_done(self, context, response):
            await memory.collect_reply(ev, response)

    provider = FakeProvider(
        [
            LLMResponse(
                role="assistant",
                tools_call_name=["lcm_search"],
                tools_call_args=[{"query": "蓝猫"}],
                tools_call_ids=["search-1"],
            ),
            LLMResponse(role="assistant", completion_text="Alice 喜欢蓝猫。"),
        ]
    )
    bridge = RunnerBridge(owner)
    try:
        loop = await runner(owner, ev, req, provider, Hooks())
        loop.tool_executor = FunctionToolExecutor()
        async for _ in loop.step():
            pass
        assert not loop.done()
        async for _ in loop.step():
            pass
        assert loop.done()
        assert memory.memory_queue.pending_count() == 2
        assert len(provider.calls) == 2
        second = json.dumps(
            [m.model_dump() for m in provider.calls[1]["contexts"]], ensure_ascii=False
        )
        assert "蓝猫" in second and old in second and "search-1" in second
        records = owner.store.expand(owner.scope, ev.get_extra("lcm_run"))["units"][0][
            "records"
        ]
        assert [n["payload"]["message"]["role"] for n in records] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        # Browsing and expansion cannot produce a new incoming event or reply.
        owner.store.search(owner.scope, "蓝猫")
        owner.store.expand(owner.scope, old)
        assert memory.memory_queue.pending_count() == 2
    finally:
        bridge.close()


@pytest.mark.asyncio
async def test_nested_agent_request_does_not_create_a_second_formal_round(owner):
    ev, req = await prepare(owner)
    bridge = RunnerBridge(owner)
    try:
        parent = await runner(owner, ev, req, FakeProvider())
        nested = await runner(
            owner, ev, ProviderRequest(prompt="nested"), FakeProvider()
        )
        assert not hasattr(nested.provider, "_owner")
        assert len(owner.store.runs(owner.scope)) == 1
        async for _ in parent.step():
            pass
    finally:
        bridge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mention", [False, True])
async def test_reply_gate_only_shortens_its_copy_and_mentions_bypass(owner, mention):
    from astrbot.api.message_components import At, Plain

    from astrbot_plugin_reply_gate.main import ReplyGate
    from astrbot_plugin_reply_gate.shared_context import MODEL

    owner.store.append(owner.scope, {"role": "user", "content": "最老原文" * 5000})
    owner.store.append(owner.scope, {"role": "user", "content": "最近聊天"})
    ev, req = await prepare(owner)
    ev.get_messages = lambda: ([At(qq="bot")] if mention else []) + [Plain("current")]
    ev.get_self_id = lambda: "bot"
    ev.stop_event = Mock()
    ev.set_extra("reply_gate_candidate", True)
    ev.set_extra("reply_gate_memory", "当前人格和长期记忆")
    original = copy.deepcopy(req.contexts)
    gate = ReplyGate(
        owner.context,
        {
            "bailian_api_key": "isolated-test-marker",
            "bailian_api_url": "https://isolated.invalid/systemone",
            "context_budget": 4000,
        },
    )
    captured = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self, **kwargs):
            return {
                "model": MODEL,
                "answers": {"should_reply": {"type": "noul", "noul": 0.99}},
            }

    def post(*args, **kwargs):
        captured.append(kwargs["json"])
        return Response()

    gate._http = SimpleNamespace(closed=False, post=post, close=AsyncMock())
    try:
        await gate.select_reply(ev, req)
        decision = ev.get_extra("reply_gate_decision")
        assert decision["reply"]
        assert req.contexts == original and "最老原文" in json.dumps(
            req.contexts, ensure_ascii=False
        )
        ev.stop_event.assert_not_called()
        if mention:
            assert decision["source"] == "mention" and captured == []
        else:
            assert decision["dropped_messages"] >= 1
            assert "最老原文" not in json.dumps(captured, ensure_ascii=False)
            assert "当前人格和长期记忆" in json.dumps(captured, ensure_ascii=False)
    finally:
        await gate.terminate()
