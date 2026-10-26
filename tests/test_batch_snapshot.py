"""Exercise serial admission, judgment and the native tool loop together offline."""

import asyncio
import copy
from types import SimpleNamespace

import pytest
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from astrbot_plugin_lcm.bridge import RunnerBridge
from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_effective_pipeline import (
    MemoryConversationManager,
    native_save,
    run_to_end,
)
from astrbot_plugin_lcm.tests.test_integration import FakeProvider, runner
from astrbot_plugin_reply_gate.main import ReplyGate
from astrbot_plugin_reply_gate.tests.test_reply_gate import make_event

owner = test_integration.owner


class DecisionStub:
    closed = False

    def __init__(self, probability=0.99, during=None):
        self.probability = probability
        self.during = during
        self.calls = []

    def post(self, endpoint, **kwargs):
        self.calls.append(copy.deepcopy(kwargs["json"]))
        return self

    async def __aenter__(self):
        if self.during:
            await self.during()
        return self

    async def __aexit__(self, *args):
        return False

    status = 200

    async def json(self, **kwargs):
        return {"answers": {"should_reply": {"type": "noul", "noul": self.probability}}}

    async def close(self):
        self.closed = True


class DuringTool(FunctionTool):
    def __init__(self, action):
        super().__init__(
            name="during_tool",
            description="Offline callback.",
            parameters={"type": "object", "properties": {}},
        )
        self.action = action

    async def call(self, context):
        await self.action()
        return "CURRENT_TOOL_RESULT"


def incoming(text, mention=False):
    ev = make_event(text, mention=mention)
    ev.message_obj.message_id = text
    return ev


def setup(owner, ev, probability=0.99, activity=0.5):
    old = [{"role": "user", "content": f"OLD_HISTORY_{i}"} for i in range(4)]
    manager = MemoryConversationManager(old)
    owner.context = SimpleNamespace(
        conversation_manager=manager,
        get_config=lambda *a, **k: {
            "timezone": "Asia/Shanghai",
            "provider_settings": {},
        },
        get_registered_star=lambda name: None,
    )
    owner.scope = owner.store.scope(ev.unified_msg_origin, "conv")
    owner.store.import_history(owner.scope, [])
    for item in old:
        owner.store.append(owner.scope, item, tokens=20)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    gate = ReplyGate(
        owner.context,
        {
            "bailian_api_key": "offline-placeholder",
            "bailian_api_url": "https://offline.invalid",
            "reply_activity": activity,
            "merge_idle_seconds": 0,
            "merge_max_wait_seconds": 0,
            "context_budget": 20000,
        },
    )
    gate._http = DecisionStub(probability)
    return manager, history, execution, gate


async def request(owner, gate, ev):
    batch = ev.get_extra("reply_gate_batch")
    assert gate._batcher.begin(batch.owner)
    # Other handlers/media assembly have separate coverage. Preserve each
    # sibling as a distinct user message while exercising real admission/save.
    batch.past_messages = [
        {"role": "user", "content": m.event.message_str}
        for m in batch.members
        if m is not batch.owner
    ]
    req = ProviderRequest(
        prompt=ev.message_str,
        contexts=copy.deepcopy(owner.context.conversation_manager.history),
        conversation=SimpleNamespace(cid="conv", token_usage=0),
        func_tool=ToolSet(),
    )
    await gate.expand_batch(ev, req)
    await owner.prepare(ev, req)
    await owner.remove_duplicate_context(ev, req)
    return req


async def registered_runner(owner, ev, req, provider):
    loop = await runner(owner, ev, req, provider)
    internal.register_active_runner(ev.unified_msg_origin, loop)
    cleanup = ev.cleanup_temporary_local_files

    def finish():
        internal.unregister_active_runner(ev.unified_msg_origin, loop)
        cleanup()

    ev.cleanup_temporary_local_files = finish
    return loop


@pytest.mark.asyncio
async def test_judge_reply_tools_share_snapshot_and_pending_events_run_once(owner):
    active = incoming("BATCH_A")
    manager, history, execution, gate = setup(owner, active)
    pending, tasks = [], []

    async def enqueue(text):
        ev = incoming(text)
        pending.append(ev)
        await owner.archive(ev)
        tasks.append(asyncio.create_task(gate.observe_message(ev)))
        await asyncio.sleep(0)
        return ev

    try:
        await owner.archive(active)
        await gate.observe_message(active)
        assert gate._batcher.begin(active.get_extra("reply_gate_batch_member"))
        await enqueue("BEFORE_JUDGMENT_B")
        req = await request(owner, gate, active)
        gate._http.during = lambda: enqueue("DURING_JUDGMENT_C")
        await gate.select_reply(active, req)
        judge = str(gate._http.calls[0])
        assert "BATCH_A" in judge and "OLD_HISTORY_0" in judge
        assert "BEFORE_JUDGMENT_B" not in judge
        state = active.get_extra("lcm_prepared")
        frozen = copy.deepcopy(state["frozen_base"])

        async def during_tool():
            await enqueue("DURING_TOOL_D")
            job_id = owner.store.plan_job(owner.scope, 40)
            job = owner.store.next_job()
            assert job and job["id"] == job_id
            assert not {ev.get_extra("lcm_input")["id"] for ev in pending}.intersection(
                job["sources"]
            )
            owner.store.job_result(job_id, "PUBLISHED_OLD_SUMMARY")
            assert owner.store.publish(job)

        req.func_tool.add_tool(DuringTool(during_tool))
        provider = FakeProvider(
            [
                LLMResponse(
                    role="assistant",
                    tools_call_name=["during_tool"],
                    tools_call_args=[{}],
                    tools_call_ids=["offline-tool"],
                ),
                LLMResponse(role="assistant", completion_text="ANSWER_TO_A"),
            ]
        )
        loop = await registered_runner(owner, active, req, provider)
        loop.tool_executor = FunctionToolExecutor()
        await run_to_end(loop)
        assert len(provider.calls) == 2
        for call in provider.calls:
            text = str(call["contexts"])
            assert "BATCH_A" in text and "OLD_HISTORY_0" in text
            assert all(ev.message_str not in text for ev in pending)
            assert "PUBLISHED_OLD_SUMMARY" not in text
        assert "CURRENT_TOOL_RESULT" in str(provider.calls[1]["contexts"])
        assert state["frozen_base"] == frozen
        assert all(not task.done() for task in tasks)
        await native_save(manager, active, req, loop)
        active.cleanup_temporary_local_files()
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        batch = pending[-1].get_extra("reply_gate_batch")
        assert [m.event for m in batch.members] == pending
        assert active not in [m.event for m in batch.members]
        for ev in pending[:-1]:
            await gate.prepare_batch(ev)
            assert ev.is_stopped()
            ev.cleanup_temporary_local_files()
        next_req = await request(owner, gate, pending[-1])
        gate._http.during = None
        gate._http.probability = 0
        await gate.select_reply(pending[-1], next_req)
        next_judge = str(gate._http.calls[1])
        assert all(ev.message_str in next_judge for ev in pending)
        assert "ANSWER_TO_A" in next_judge
        assert "PUBLISHED_OLD_SUMMARY" in next_judge
        assert pending[-1].is_stopped()
        pending[-1].cleanup_temporary_local_files()
        await asyncio.sleep(0)
        assert len(gate._http.calls) == 2
        assert not gate._batcher._active
    finally:
        for ev in [active, *pending]:
            ev.cleanup_temporary_local_files()
        await gate.terminate()
        await asyncio.gather(*tasks, return_exceptions=True)
        execution.close()
        history.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mention,activity,probability,reply",
    [
        (False, 0.5, 0, False),
        (True, 0, 0, True),
        (False, 1, 0, True),
    ],
)
async def test_silence_and_bypass_paths_claim_once_and_pin_context(
    owner, mention, activity, probability, reply
):
    active = incoming("CURRENT", mention)
    manager, history, execution, gate = setup(owner, active, probability, activity)
    later = incoming("NEXT_MESSAGE")
    task = None
    try:
        await owner.archive(active)
        await gate.observe_message(active)
        req = await request(owner, gate, active)
        await gate.select_reply(active, req)
        assert active.get_extra("reply_gate_decision")["reply"] is reply
        assert active.get_extra("lcm_prepared")["frozen_base"] is not None
        await owner.archive(later)
        task = asyncio.create_task(gate.observe_message(later))
        await asyncio.sleep(0)
        assert not task.done()
        if reply:
            provider = FakeProvider()
            loop = await registered_runner(owner, active, req, provider)
            await run_to_end(loop)
            assert "NEXT_MESSAGE" not in str(provider.calls)
            await native_save(manager, active, req, loop)
        else:
            assert active.is_stopped()
        active.cleanup_temporary_local_files()
        await asyncio.wait_for(task, 1)
        assert [m.event for m in later.get_extra("reply_gate_batch").members] == [later]
        later.cleanup_temporary_local_files()
        assert not gate._batcher._active
    finally:
        active.cleanup_temporary_local_files()
        later.cleanup_temporary_local_files()
        await gate.terminate()
        if task:
            await asyncio.gather(task, return_exceptions=True)
        execution.close()
        history.close()


@pytest.mark.asyncio
async def test_archive_hold_protects_waiting_message_and_cleanup_releases_it(owner):
    ev = incoming("WAITING_ARCHIVE")
    _, history, execution, gate = setup(owner, ev)
    try:
        await owner.archive(ev)
        ident = ev.get_extra("lcm_input")["id"]
        hold = ev.get_extra("lcm_pending_owner")
        assert owner.store.node(owner.scope, ident)["reservation"] == hold
        assert ident in {n["id"] for n in owner.store.view(owner.scope)["nodes"]}
        assert ident not in {
            n["id"] for n in owner.view(owner.scope, include_pending=False)["nodes"]
        }
        with pytest.raises(ValueError, match="reserved"):
            owner.store.reserve(owner.scope, [ident], "unrelated")
        owner.store.reserve(owner.scope, [ident], "claimed", previous_owners=[hold])
        ev.cleanup_temporary_local_files()
        assert owner.store.node(owner.scope, ident)["reservation"] == "claimed"
        owner.store.release("claimed")
        assert owner.store.node(owner.scope, ident)["reservation"] is None
    finally:
        ev.cleanup_temporary_local_files()
        await gate.terminate()
        execution.close()
        history.close()
