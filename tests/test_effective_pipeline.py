"""Check the actual runner, native serializer, and unified history commit path."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import ImageURLPart, TextPart
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from astrbot_plugin_lcm.bridge import RunnerBridge
from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.history_model import safe_history
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_integration import (
    FakeProvider,
    event,
    prepare,
    runner,
)
from astrbot_plugin_reply_gate.parallel_runtime import ParallelRuntime

owner = test_integration.owner

MIME_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/ZXcAAAAASUVORK5CYII="
PRIVATE = "PRIVATE_MEME_SEARCH_AND_SELECTION"


class MemoryConversationManager:
    """Persist the exact native payload without replacing its serializer."""

    def __init__(self, history=()):
        self.history = copy.deepcopy(list(history))
        self.writes = []

    async def get_curr_conversation_id(self, unified_msg_origin):
        return "conv"

    async def get_conversation(self, unified_msg_origin, conversation_id=None):
        return SimpleNamespace(
            cid="conv", history=json.dumps(self.history), token_usage=0
        )

    async def update_conversation(
        self,
        unified_msg_origin,
        conversation_id=None,
        history=None,
        title=None,
        persona_id=None,
        token_usage=None,
    ):
        if history is not None:
            self.history = copy.deepcopy(history)
            self.writes.append(copy.deepcopy(history))


class ChooseMeme(FunctionTool):
    def __init__(self):
        super().__init__(
            name="choose_meme",
            description="Return an isolated meme search result.",
            parameters={"type": "object", "properties": {}},
        )

    async def call(self, context):
        return PRIVATE * 100


class SavedImageHooks(BaseAgentRunHooks):
    def __init__(self, hide_tools):
        self.hide_tools = hide_tools

    async def on_agent_done(self, context, response):
        # Stand in for another plugin recording its successful image send.
        # The LCM layer sees only normal native message/save conventions.
        final = context.messages[-1]
        final.content.append(
            ImageURLPart(image_url=ImageURLPart.ImageURL(url=MIME_IMAGE))
        )
        final.content.append(
            TextPart(text="PRIVATE_EPHEMERAL_OUTPUT_PART").mark_as_temp()
        )
        if self.hide_tools:
            for item in context.messages:
                if item.tool_calls or item.role == "tool":
                    item._no_save = True


async def run_to_end(loop):
    while not loop.done():
        async for _ in loop.step():
            pass


async def native_save(manager, event, req, loop):
    await InternalAgentSubStage._save_to_history(
        SimpleNamespace(conv_manager=manager),
        event,
        req,
        loop.get_final_llm_resp(),
        loop.run_context.messages,
        None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("hide_with", ["no_save", "history_edit"])
async def test_real_tool_loop_follows_native_visibility_and_compresses_saved_image(
    owner, hide_with
):
    manager = MemoryConversationManager()
    owner.context = SimpleNamespace(conversation_manager=manager)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    try:
        ev, req = await prepare(owner, "send an appropriate meme")
        req.func_tool.add_tool(ChooseMeme())
        provider = FakeProvider(
            [
                LLMResponse(
                    role="assistant",
                    tools_call_name=["choose_meme"],
                    tools_call_args=[{}],
                    tools_call_ids=["meme-search"],
                ),
                LLMResponse(role="assistant", completion_text="这是刚刚发出的表情图。"),
            ]
        )
        loop = await runner(
            owner, ev, req, provider, SavedImageHooks(hide_with == "no_save")
        )
        loop.tool_executor = FunctionToolExecutor()
        await run_to_end(loop)
        state = ev.get_extra("lcm_prepared")
        run = state["run"]
        assert loop.done() and len(provider.calls) == 2
        assert PRIVATE in str(provider.calls[1]["contexts"])
        raw_snapshot = [item.model_dump() for item in loop.run_context.messages]
        archived = owner.store.expand(owner.scope, run)["units"][0]["records"]
        assert [item["payload"]["message"]["role"] for item in archived] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert PRIVATE in str(archived)
        assert "PRIVATE_EPHEMERAL_OUTPUT_PART" in str(archived)
        active = {item["id"] for item in owner.store.view(owner.scope)["nodes"]}
        assert not active.intersection(state["output_ids"])
        assert owner.store.runs(owner.scope)[0]["committed"] == 0
        owner.store.append(
            owner.scope,
            {"role": "user", "content": "newer protected content"},
            tokens=100,
        )
        assert owner.store.plan_job(owner.scope, 40) is None

        await native_save(manager, ev, req, loop)
        assert owner.store.runs(owner.scope)[0]["committed"] == 1
        if hide_with == "history_edit":
            saved = [
                item
                for item in safe_history(manager.history)
                if not item.get("tool_calls") and item.get("role") != "tool"
            ]
            await manager.update_conversation("group", "conv", history=saved)

        effective = owner.messages(owner.store.view(owner.scope)["nodes"])
        assert PRIVATE not in str(effective)
        assert "PRIVATE_EPHEMERAL_OUTPUT_PART" not in str(effective)
        assert MIME_IMAGE in str(effective) and "这是刚刚发出的表情图" in str(effective)
        assert not any(
            item.get("tool_calls") or item.get("role") == "tool" for item in effective
        )
        _, next_req = await prepare(owner, "what did you send")
        assert PRIVATE not in str(next_req.contexts)
        assert "PRIVATE_EPHEMERAL_OUTPUT_PART" not in str(next_req.contexts)
        assert MIME_IMAGE in str(next_req.contexts)

        job_id = owner.store.plan_job(owner.scope, 40)
        job = owner.store.next_job()
        assert job and job["id"] == job_id
        sources = owner.messages(
            [owner.store.node(owner.scope, ident) for ident in job["sources"]]
        )
        assert PRIVATE not in str(
            sources
        ) and "PRIVATE_EPHEMERAL_OUTPUT_PART" not in str(sources)
        assert MIME_IMAGE in str(sources)
        owner.store.job_result(job_id, "机器人发了一张表情图并说明已发送。")
        assert owner.store.publish(job)
        assert [item.model_dump() for item in loop.run_context.messages] == raw_snapshot
        raw_again = owner.store.expand(owner.scope, run)["units"][0]["records"]
        assert PRIVATE in str(raw_again) and "PRIVATE_EPHEMERAL_OUTPUT_PART" in str(
            raw_again
        )
        ev.cleanup_temporary_local_files()
    finally:
        execution.close()
        history.close()


@pytest.mark.asyncio
async def test_running_round_keeps_snapshot_when_unified_history_deletes_old_message(
    owner,
):
    original = {"role": "user", "content": "old source to delete"}
    owner.store.append(owner.scope, original)
    manager = MemoryConversationManager([original])
    owner.context = SimpleNamespace(conversation_manager=manager)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    try:
        ev, req = await prepare(owner, "running request", base=[original])
        pause = asyncio.Event()
        provider = FakeProvider(pause=pause)
        loop = await runner(owner, ev, req, provider)
        initial = [item.model_dump() for item in loop.run_context.messages]
        task = asyncio.create_task(run_to_end(loop))
        await asyncio.sleep(0.02)
        await manager.update_conversation("group", "conv", history=[])
        assert [item.model_dump() for item in loop.run_context.messages] == initial
        assert "old source to delete" not in str(
            owner.messages(owner.store.view(owner.scope)["nodes"])
        )
        pause.set()
        await task
        await native_save(manager, ev, req, loop)
        assert "old source to delete" not in str(manager.history)
        assert "running request" in str(manager.history)
        effective = owner.messages(owner.store.view(owner.scope)["nodes"])
        assert "old source to delete" not in str(effective)
        assert "最终回复" in str(effective)
        ev.cleanup_temporary_local_files()
    finally:
        execution.close()
        history.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("load_order", ["gate_first", "history_first"])
async def test_real_parallel_reply_gate_and_lcm_keep_both_rounds_and_external_edits(
    owner, load_order
):
    initial = [
        {"role": "user", "content": "old message to delete"},
        {"role": "user", "content": "old message to correct"},
    ]
    for item in initial:
        owner.store.append(owner.scope, item)
    manager = MemoryConversationManager(initial)
    owner.context = SimpleNamespace(conversation_manager=manager)
    gate = ParallelRuntime(owner.context)
    if load_order == "gate_first":
        bootstrap = event("bootstrap")
        gate.enter(bootstrap)
        bootstrap.cleanup_temporary_local_files()
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    release_a, ready_a, ready_b = asyncio.Event(), asyncio.Event(), asyncio.Event()
    tasks = []

    async def wait_ready(task, ready):
        waiter = asyncio.create_task(ready.wait())
        try:
            done, _ = await asyncio.wait(
                [task, waiter], timeout=2, return_when=asyncio.FIRST_COMPLETED
            )
            assert done, "parallel worker did not reach runner reset"
            if task.done():
                await task
            await waiter
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def worker(label, ready, pause=None):
        class HashableEvent(SimpleNamespace):
            __hash__ = object.__hash__
            __eq__ = object.__eq__

        ev = HashableEvent(**vars(event(label)))
        gate.enter(ev)
        history.bind(ev)
        loop = None
        try:
            async with internal.session_lock_manager.acquire_lock("group"):
                ident = owner.store.append(
                    owner.scope,
                    {"role": "user", "content": "input " + label},
                    source=label,
                )
                ev.set_extra("lcm_input", {"scope": owner.scope, "id": ident})
                req = ProviderRequest(
                    prompt="input " + label,
                    contexts=copy.deepcopy(manager.history),
                    system_prompt="persona",
                    conversation=SimpleNamespace(cid="conv", token_usage=0),
                )
                ev.set_extra("provider_request", req)
                await gate.prepare(ev, req, len(req.contexts))
                await owner.prepare(ev, req)
                loop = await runner(
                    owner,
                    ev,
                    req,
                    FakeProvider(
                        [
                            LLMResponse(
                                role="assistant", completion_text="answer " + label
                            )
                        ],
                        pause=pause,
                    ),
                )
                internal.register_active_runner("group", loop)
                ready.set()
                await run_to_end(loop)
                await native_save(manager, ev, req, loop)
                await native_save(manager, ev, req, loop)
                return ev.get_extra("lcm_run")
        finally:
            if loop is not None:
                internal.unregister_active_runner("group", loop)
            ev.cleanup_temporary_local_files()

    try:
        a = asyncio.create_task(worker("A", ready_a, release_a))
        tasks.append(a)
        await wait_ready(a, ready_a)
        b = asyncio.create_task(worker("B", ready_b))
        tasks.append(b)
        await wait_ready(b, ready_b)
        run_b = await asyncio.wait_for(b, 2)
        assert not a.done()
        updated = [
            {"role": "user", "content": "corrected old message"}
        ] + copy.deepcopy(manager.history[2:])
        await manager.update_conversation("group", "conv", history=updated)
        release_a.set()
        run_a = await asyncio.wait_for(a, 2)
        native = str(manager.history)
        visible = str(owner.messages(owner.store.view(owner.scope)["nodes"]))
        for data in (native, visible):
            assert (
                "old message to delete" not in data
                and "old message to correct" not in data
            )
            assert "corrected old message" in data
            for label in ("A", "B"):
                assert data.count("input " + label) == 1
                assert data.count("answer " + label) == 1
        for run in (run_a, run_b):
            stored = next(
                item for item in owner.store.runs(owner.scope) if item["id"] == run
            )
            assert stored["status"] == "complete" and stored["committed"] == 1
    finally:
        release_a.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        execution.close()
        if load_order == "gate_first":
            history.close()
            gate.close()
        else:
            gate.close()
            history.close()


@pytest.mark.asyncio
async def test_nested_native_save_does_not_commit_the_parent_lcm_round(owner):
    manager = MemoryConversationManager()
    owner.context = SimpleNamespace(conversation_manager=manager)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    try:
        ev, parent_req = await prepare(owner, "parent request")
        parent = await runner(owner, ev, parent_req, FakeProvider())
        parent_run = ev.get_extra("lcm_run")
        child_req = ProviderRequest(
            prompt="nested request",
            conversation=SimpleNamespace(cid="conv", token_usage=0),
        )
        child = await runner(
            owner,
            ev,
            child_req,
            FakeProvider(
                [LLMResponse(role="assistant", completion_text="nested answer")]
            ),
        )
        await run_to_end(child)
        await native_save(manager, ev, child_req, child)
        assert len(owner.store.runs(owner.scope)) == 1
        assert owner.store.runs(owner.scope)[0]["committed"] == 0
        assert owner.store.runs(owner.scope)[0]["id"] == parent_run
        await run_to_end(parent)
        await native_save(manager, ev, parent_req, parent)
        assert owner.store.runs(owner.scope)[0]["committed"] == 1
        ev.cleanup_temporary_local_files()
    finally:
        execution.close()
        history.close()


@pytest.mark.asyncio
async def test_repeated_unadmitted_input_survives_deletion_of_identical_old_input(
    owner,
):
    original = {"role": "user", "content": "same question"}
    old = owner.store.append(owner.scope, original)
    manager = MemoryConversationManager([original])
    owner.context = SimpleNamespace(conversation_manager=manager)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    try:
        ev, req = await prepare(owner, "same question", base=[original])
        req.extra_user_content_parts = []
        loop = await runner(owner, ev, req, FakeProvider())
        current = ev.get_extra("lcm_input")["id"]
        await manager.update_conversation("group", "conv", history=[])
        await run_to_end(loop)
        await native_save(manager, ev, req, loop)
        assert str(manager.history).count("same question") == 1
        ids = {node["id"] for node in owner.store.view(owner.scope)["nodes"]}
        assert old not in ids and current in ids
        ev.cleanup_temporary_local_files()
    finally:
        execution.close()
        history.close()


@pytest.mark.asyncio
async def test_native_temporary_input_part_is_not_reintroduced_from_raw_archive(owner):
    class HideInputPart(BaseAgentRunHooks):
        async def on_agent_done(self, context, response):
            incoming = next(item for item in context.messages if item.role == "user")
            incoming.content[0].mark_as_temp()

    manager = MemoryConversationManager()
    owner.context = SimpleNamespace(conversation_manager=manager)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    try:
        ev, req = await prepare(owner, "PRIVATE_INPUT_PART")
        loop = await runner(owner, ev, req, FakeProvider(), HideInputPart())
        await run_to_end(loop)
        await native_save(manager, ev, req, loop)
        assert "PRIVATE_INPUT_PART" not in str(manager.history)
        assert "PRIVATE_INPUT_PART" not in str(
            owner.messages(owner.store.view(owner.scope)["nodes"])
        )
        raw = owner.store.node(owner.scope, ev.get_extra("lcm_input")["id"])
        assert "PRIVATE_INPUT_PART" in str(raw["payload"])
        ev.cleanup_temporary_local_files()
    finally:
        execution.close()
        history.close()


@pytest.mark.asyncio
async def test_each_tool_step_uses_latest_summary_and_other_committed_reply(owner):
    original = [
        {"role": "user", "content": "old source question " * 1000},
        {"role": "assistant", "content": "old source answer " * 1000},
    ]
    for value in original:
        owner.store.append(owner.scope, value, tokens=1000)
    manager = MemoryConversationManager(original)
    owner.context = SimpleNamespace(conversation_manager=manager)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    try:
        ev, req = await prepare(owner, "current question", base=original)
        req.func_tool.add_tool(ChooseMeme())
        provider = FakeProvider(
            [
                LLMResponse(
                    role="assistant",
                    tools_call_name=["choose_meme"],
                    tools_call_args=[{}],
                    tools_call_ids=["private-lookup"],
                ),
                LLMResponse(role="assistant", completion_text="current answer"),
            ]
        )
        loop = await runner(owner, ev, req, provider)
        loop.tool_executor = FunctionToolExecutor()
        async for _ in loop.step():
            pass
        job_id = owner.store.plan_job(owner.scope, 100)
        job = owner.store.next_job()
        assert job_id and job["id"] == job_id
        owner.store.job_result(job_id, "LATEST_COMPACTED_SUMMARY")
        assert owner.store.publish(job)
        await manager.update_conversation(
            "group",
            "conv",
            history=original
            + [{"role": "assistant", "content": "OTHER_ROUND_FINISHED"}],
        )
        await run_to_end(loop)
        second = str(provider.calls[-1]["contexts"])
        assert "LATEST_COMPACTED_SUMMARY" in second
        assert "OTHER_ROUND_FINISHED" in second
        assert "old source question" not in second
        assert PRIVATE in second
        assert second.count("current question") == 1
        await native_save(manager, ev, req, loop)
        assert "OTHER_ROUND_FINISHED" in str(manager.history)
        assert "current answer" in str(manager.history)
        ev.cleanup_temporary_local_files()
    finally:
        execution.close()
        history.close()
