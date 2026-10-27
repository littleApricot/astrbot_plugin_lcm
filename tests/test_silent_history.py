"""Exercise silent finish, official persistence and compression together offline."""

import copy
import json
from types import SimpleNamespace

import pytest
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
from astrbot.core.provider.entities import LLMResponse

from astrbot_plugin_lcm.bridge import RunnerBridge
from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_effective_pipeline import (
    MemoryConversationManager,
    native_save,
    run_to_end,
)
from astrbot_plugin_lcm.tests.test_integration import FakeProvider, prepare, runner
from astrbot_plugin_silence.main import Silence
from astrbot_plugin_silence.tests.test_history import call, result

owner = test_integration.owner


class PersistedManager(MemoryConversationManager):
    def __init__(self):
        super().__init__()

        async def write(cid, content=None, **kwargs):
            if content is not None:
                self.history = copy.deepcopy(content)
                self.writes.append(copy.deepcopy(content))

        self.db = SimpleNamespace(update_conversation=write)

    async def update_conversation(
        self,
        unified_msg_origin,
        conversation_id=None,
        history=None,
        title=None,
        persona_id=None,
        token_usage=None,
    ):
        await self.db.update_conversation(conversation_id, content=history)


class Finish(FunctionTool):
    def __init__(self, plugin):
        super().__init__(
            name="silence",
            description="Finish the offline round.",
            parameters={"type": "object", "properties": {}},
        )
        self.plugin = plugin

    async def call(self, context):
        return await self.plugin.silence(context.context.event)


class PreviousAction(FunctionTool):
    def __init__(self):
        super().__init__(
            name="previous_action",
            description="Offline action.",
            parameters={"type": "object", "properties": {}},
        )

    async def call(self, context):
        return "CURRENT_ACTION_RESULT"


@pytest.mark.asyncio
async def test_silence_is_complete_and_absent_from_saved_reply_and_compression(
    owner, monkeypatch
):
    class Event(SimpleNamespace):
        __hash__ = object.__hash__

    original_event = test_integration.event
    monkeypatch.setattr(
        test_integration, "event", lambda text: Event(**original_event(text).__dict__)
    )
    manager = PersistedManager()
    context = SimpleNamespace(conversation_manager=manager)
    plugin = Silence(context)
    context.get_registered_star = lambda name: (
        SimpleNamespace(star_cls=plugin, activated=True)
        if name == "astrbot_plugin_silence"
        else None
    )
    owner.context = context
    previous = [
        {"role": "user", "content": "之前的问题"},
        call("edit_memory", "memory"),
        result("memory"),
        {"role": "assistant", "content": "[机器人已发送语音] 好嘛"},
        {"role": "user", "content": "[历史记录：机器人已发送图片] image-receipt"},
    ]
    await manager.update_conversation("group", "conv", history=previous)
    owner.store.sync_history(owner.scope, [], previous)
    history = HistoryBridge(owner)
    owner.history_bridge = history
    execution = RunnerBridge(owner)
    ev, loop = None, None
    try:
        ev, req = await prepare(owner, "这句不用回")
        req.func_tool.add_tool(PreviousAction())
        req.func_tool.add_tool(Finish(plugin))
        provider = FakeProvider(
            [
                LLMResponse(
                    role="assistant",
                    tools_call_name=["previous_action"],
                    tools_call_args=[{}],
                    tools_call_ids=["previous-action"],
                ),
                LLMResponse(
                    role="assistant",
                    tools_call_name=["silence"],
                    tools_call_args=[{}],
                    tools_call_ids=["silent"],
                    reasoning_content="SILENT_STEP_REASONING",
                ),
            ]
        )
        loop = await runner(owner, ev, req, provider)
        loop.tool_executor = FunctionToolExecutor()
        internal.register_active_runner(ev.unified_msg_origin, loop)
        await run_to_end(loop)
        assert len(provider.calls) == 2
        assert owner.store.runs(owner.scope)[0]["status"] == "complete"
        await native_save(manager, ev, req, loop)
        assert "SILENT_STEP_REASONING" not in str(manager.history)
        assert '"silence"' not in json.dumps(manager.history)
        assert "memory" in str(manager.history)
        assert "image-receipt" in str(manager.history)
        assert "好嘛" in str(manager.history)
        assert "这句不用回" in str(manager.history)
        assert "CURRENT_ACTION_RESULT" in str(manager.history)
        assert owner.store.runs(owner.scope)[0]["committed"] == 1
        next_ev, next_req = await prepare(owner, "之后的新消息")
        assert "SILENT_STEP_REASONING" not in str(next_req.contexts)
        assert "image-receipt" in str(next_req.contexts)
        assert "好嘛" in str(next_req.contexts)
        assert "CURRENT_ACTION_RESULT" in str(next_req.contexts)
        assert "状态=failed" not in str(next_req.contexts)
        compression = owner.messages(owner.compression_nodes(owner.scope))
        assert "SILENT_STEP_REASONING" not in str(compression)
        assert "image-receipt" in str(compression)
        assert "好嘛" in str(compression)
        assert "CURRENT_ACTION_RESULT" in str(compression)
        next_ev.cleanup_temporary_local_files()
    finally:
        if ev and loop:
            internal.unregister_active_runner(ev.unified_msg_origin, loop)
            ev.cleanup_temporary_local_files()
        execution.close()
        history.close()
        await plugin.terminate()


def test_legacy_failed_silence_is_removed_before_lcm_wraps_or_compresses(owner):
    plugin = Silence(SimpleNamespace())
    owner.context = SimpleNamespace(
        get_all_stars=lambda: [SimpleNamespace(star_cls=plugin)]
    )
    run = owner.store.begin(owner.scope, "old", owner.store.view(owner.scope), [])
    for message in [call(), result()]:
        owner.store.append(owner.scope, message, round_id=run)
    owner.store.finish(run, "failed")
    assert owner.messages(owner.store.view(owner.scope)["nodes"]) == []
    assert owner.messages(owner.compression_nodes(owner.scope)) == []
    # Debug originals stay intact outside the active context.
    assert len(owner.store.expand(owner.scope, run)["units"][0]["records"]) == 2
