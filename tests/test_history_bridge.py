"""Exercise committed-history observation without model or platform calls."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from astrbot.core.agent.message import Message, TextPart
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.provider.entities import LLMResponse, ProviderRequest

from astrbot_plugin_lcm.history_bridge import HistoryBridge, merge_branch
from astrbot_plugin_lcm.store import Store


def message(text, role="user"):
    return {"role": role, "content": text}


class NativeManager:
    def __init__(self):
        self.histories = {}
        self.calls = []
        self.fail = False

    async def get_curr_conversation_id(self, unified_msg_origin):
        return "conversation"

    async def get_conversation(self, unified_msg_origin, conversation_id):
        return SimpleNamespace(
            history=json.dumps(
                self.histories.get((unified_msg_origin, conversation_id), [])
            )
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
        if self.fail:
            raise RuntimeError("native write failed")
        self.calls.append((unified_msg_origin, conversation_id, copy.deepcopy(history)))
        if history is not None:
            conversation_id = conversation_id or "conversation"
            self.histories[(unified_msg_origin, conversation_id)] = copy.deepcopy(
                history
            )


@pytest.fixture
def bridge(tmp_path):
    manager = NativeManager()
    owner = SimpleNamespace(
        store=Store(tmp_path / "history.db"),
        context=SimpleNamespace(conversation_manager=manager),
        wake=asyncio.Event(),
        messages=lambda nodes: [
            copy.deepcopy(node["payload"]["message"]) for node in nodes
        ],
    )
    observer = HistoryBridge(owner)
    try:
        yield observer
    finally:
        observer.close()


async def seed(bridge, texts=("original",), origin="group"):
    scope = bridge.owner.store.scope(origin, "conversation")
    values = [message(text) for text in texts]
    ids = [bridge.owner.store.append(scope, value) for value in values]
    bridge.manager.histories[(origin, "conversation")] = copy.deepcopy(values)
    await bridge.refresh(origin, "conversation")
    return scope, ids


def projected(bridge, scope):
    return bridge.owner.messages(bridge.owner.store.view(scope)["nodes"])


async def registered_round(bridge, current):
    scope = bridge.owner.store.scope("group", "conversation")
    await bridge.refresh("group", "conversation")
    store = bridge.owner.store
    input_id = store.append(scope, current.model_dump(mode="json"))
    store.reserve(scope, [input_id], "reservation")
    base = store.view(scope, exclude=[input_id])
    run = store.begin(scope, "reservation", base, [input_id])
    request = ProviderRequest(
        prompt="current input",
        conversation=SimpleNamespace(cid="conversation", token_usage=0),
    )
    runtime = [Message(role="system", content="persona"), current]
    state = {
        "owner": bridge.owner,
        "scope": scope,
        "run": run,
        "origin": "group",
        "cid": "conversation",
        "request": request,
        "base": base,
        "inputs": [input_id],
        "runner": SimpleNamespace(run_context=SimpleNamespace(messages=runtime)),
    }
    extras = {"lcm_prepared": state}
    event = SimpleNamespace(
        unified_msg_origin="group",
        get_extra=lambda key, default=None: extras.get(key, default),
    )
    await bridge.register(event, state)
    return event, state, runtime, input_id


@pytest.mark.asyncio
async def test_metadata_write_does_not_touch_history_or_sidecar(bridge):
    scope, _ = await seed(bridge)
    before = bridge.owner.store.native_history(scope)
    version = bridge.owner.store.view(scope)["version"]
    await bridge.manager.update_conversation(
        "group", "conversation", history=None, title="renamed", token_usage=12
    )
    assert bridge.owner.store.native_history(scope) == before
    assert bridge.owner.store.view(scope)["version"] == version
    assert bridge.manager.calls[-1] == ("group", "conversation", None)


@pytest.mark.asyncio
async def test_failed_native_write_cannot_commit_projection(bridge):
    scope, ids = await seed(bridge)
    baseline = bridge.owner.store.native_history(scope)
    bridge.manager.fail = True
    with pytest.raises(RuntimeError, match="native write failed"):
        await bridge.manager.update_conversation(
            "group", "conversation", history=[message("uncommitted")]
        )
    assert projected(bridge, scope) == [message("original")]
    assert bridge.owner.store.native_history(scope) == baseline
    assert bridge.owner.store.node(scope, ids[0])["payload"]["message"] == message(
        "original"
    )


@pytest.mark.asyncio
async def test_external_history_replacement_preserves_immutable_original(bridge):
    scope, ids = await seed(bridge)
    original = copy.deepcopy(bridge.owner.store.node(scope, ids[0])["payload"])
    await bridge.manager.update_conversation(
        "group", "conversation", history=[message("corrected", "assistant")]
    )
    assert projected(bridge, scope) == [message("corrected", "assistant")]
    assert bridge.owner.store.node(scope, ids[0])["payload"] == original
    assert not bridge.owner.store.node(scope, ids[0])["active"]
    assert bridge.owner.store.native_history(scope)["messages"] == [
        message("corrected", "assistant")
    ]


@pytest.mark.asyncio
async def test_refresh_repairs_successful_native_write_after_sync_failure(bridge):
    scope, ids = await seed(bridge)
    sync = bridge.sync

    def broken_sync(*args, **kwargs):
        raise RuntimeError("isolated sidecar failure")

    bridge.sync = broken_sync
    try:
        await bridge.manager.update_conversation(
            "group", "conversation", history=[message("committed replacement")]
        )
    finally:
        bridge.sync = sync
    assert bridge.manager.histories[("group", "conversation")] == [
        message("committed replacement")
    ]
    assert projected(bridge, scope) == [message("original")]
    await bridge.refresh("group", "conversation")
    assert projected(bridge, scope) == [message("committed replacement")]
    assert bridge.owner.store.node(scope, ids[0])["payload"]["message"] == message(
        "original"
    )


@pytest.mark.asyncio
async def test_empty_history_keeps_unpublished_silent_messages(bridge):
    scope, ids = await seed(bridge)
    silent = bridge.owner.store.append(scope, message("unanswered group message"))
    await bridge.manager.update_conversation("group", "conversation", history=[])
    assert projected(bridge, scope) == [message("unanswered group message")]
    assert bridge.owner.store.node(scope, silent)["active"]
    assert not bridge.owner.store.node(scope, ids[0])["active"]
    assert bridge.owner.store.native_history(scope)["messages"] == []


@pytest.mark.asyncio
async def test_history_edits_are_scoped_by_origin_and_conversation(bridge):
    first, _ = await seed(bridge, origin="first")
    second, _ = await seed(bridge, origin="second")
    second_sidecar = bridge.owner.store.native_history(second)
    await bridge.manager.update_conversation("first", history=[message("first edited")])
    assert projected(bridge, first) == [message("first edited")]
    assert projected(bridge, second) == [message("original")]
    assert bridge.owner.store.native_history(second) == second_sidecar


@pytest.mark.asyncio
async def test_native_save_respects_no_save_messages_and_temporary_parts(bridge):
    scope = bridge.owner.store.scope("group", "conversation")
    extras = {}
    event = SimpleNamespace(
        unified_msg_origin="group",
        get_extra=lambda key, default=None: extras.get(key, default),
    )
    request = ProviderRequest(
        prompt="question",
        conversation=SimpleNamespace(cid="conversation", token_usage=0),
    )
    private = Message(role="assistant", content="private search planning")
    private._no_save = True
    reply = Message(
        role="assistant",
        content=[
            TextPart(text="sent an image"),
            TextPart(text="private note").mark_as_temp(),
        ],
    )
    await InternalAgentSubStage._save_to_history(
        SimpleNamespace(conv_manager=bridge.manager),
        event,
        request,
        LLMResponse(role="assistant", completion_text="sent an image"),
        [
            Message(role="system", content="persona"),
            Message(role="user", content="question"),
            private,
            reply,
        ],
        None,
    )
    native = bridge.manager.histories[("group", "conversation")]
    assert "private" not in json.dumps(native)
    assert "persona" not in json.dumps(native)
    assert "sent an image" in json.dumps(projected(bridge, scope))
    assert "private" not in json.dumps(projected(bridge, scope))
    assert bridge.saving.get() is None


def test_close_restores_owned_wrappers(bridge):
    bridge.close()
    assert bridge.manager.update_conversation == bridge.original
    assert InternalAgentSubStage._save_to_history is bridge.save_original


@pytest.mark.asyncio
async def test_close_preserves_later_wrapper_and_disables_its_own_observer(bridge):
    observer = bridge.manager.update_conversation

    async def later(*args, **kwargs):
        return await observer(*args, **kwargs)

    bridge.manager.update_conversation = later
    original_save_wrapper = InternalAgentSubStage._save_to_history

    async def later_save(*args, **kwargs):
        return await original_save_wrapper(*args, **kwargs)

    InternalAgentSubStage._save_to_history = later_save
    try:
        bridge.close()
        assert bridge.manager.update_conversation is later
        assert InternalAgentSubStage._save_to_history is later_save
        await bridge.manager.update_conversation("group", "conversation", history=[])
        assert bridge.owner.store.scopes() == []
    finally:
        InternalAgentSubStage._save_to_history = bridge.save_original


def test_merge_branch_preserves_parallel_tail_while_replacing_old_anchor():
    original = [message("old"), message("question")]
    latest = original + [
        message("parallel input"),
        message("parallel answer", "assistant"),
    ]
    branch = [
        message("edited"),
        message("question"),
        message("own answer", "assistant"),
    ]
    assert merge_branch(original, branch, latest) == [
        message("edited"),
        message("question"),
        message("parallel input"),
        message("parallel answer", "assistant"),
        message("own answer", "assistant"),
    ]


def test_merge_branch_does_not_overwrite_a_concurrent_anchor_edit():
    initial = [message("old")]
    latest = [message("newer edit"), message("parallel tail")]
    branch = [message("stale edit"), message("own answer", "assistant")]
    assert merge_branch(initial, branch, latest, owned=[branch[-1]]) == latest + [
        message("own answer", "assistant")
    ]


@pytest.mark.asyncio
async def test_native_save_keeps_owned_reply_when_old_anchor_was_edited(bridge):
    scope, _ = await seed(bridge, texts=("old",))
    store = bridge.owner.store
    current_id = store.append(scope, message("temporary current input"))
    store.reserve(scope, [current_id], "reservation")
    base = store.view(scope, exclude=[current_id])
    run = store.begin(scope, "reservation", base, [current_id])
    request = ProviderRequest(
        prompt="temporary current input",
        conversation=SimpleNamespace(cid="conversation", token_usage=0),
    )
    old = Message(role="user", content="old")
    current = Message(role="user", content="temporary current input")
    current._no_save = True
    runtime = [Message(role="system", content="persona"), old, current]
    state = {
        "owner": bridge.owner,
        "scope": scope,
        "run": run,
        "origin": "group",
        "cid": "conversation",
        "request": request,
        "base": base,
        "inputs": [current_id],
        "runner": SimpleNamespace(run_context=SimpleNamespace(messages=runtime)),
    }
    extras = {"lcm_prepared": state}
    event = SimpleNamespace(
        unified_msg_origin="group",
        get_extra=lambda key, default=None: extras.get(key, default),
    )
    await bridge.register(event, state)
    latest = [message("newer edit"), message("parallel tail")]
    await bridge.manager.update_conversation("group", "conversation", history=latest)
    old.content = "stale edit"
    reply = Message(role="assistant", content="own answer")
    runtime.append(reply)
    reply_id = store.append(scope, reply.model_dump(mode="json"), round_id=run)
    state["output_ids"].append(reply_id)
    state["output_messages"].append((reply_id, reply))
    store.finish(run, activate=False)
    await InternalAgentSubStage._save_to_history(
        SimpleNamespace(conv_manager=bridge.manager),
        event,
        request,
        LLMResponse(role="assistant", completion_text="own answer"),
        runtime,
        None,
    )
    saved = bridge.manager.histories[("group", "conversation")]
    assert [item["content"] for item in saved] == [
        "newer edit",
        "parallel tail",
        "own answer",
    ]
    assert "stale edit" not in json.dumps(saved)
    assert "temporary current input" not in json.dumps(saved)
    assert state["history_committed"]
    assert "own answer" in json.dumps(projected(bridge, scope))


def test_merge_branch_preserves_native_checkpoints():
    checkpoint = {"role": "_checkpoint", "content": {"id": "cp-1"}}
    original = [message("old")]
    latest = original + [checkpoint, message("parallel")]
    branch = original + [message("own answer", "assistant")]
    assert merge_branch(original, branch, latest) == latest + [
        message("own answer", "assistant")
    ]


@pytest.mark.asyncio
async def test_restart_replays_committed_output_and_its_private_part_revision(bridge):
    event, state, runtime, _ = await registered_round(
        bridge, Message(role="user", content="current input")
    )
    reply = Message(
        role="assistant",
        content=[
            TextPart(text="public reply"),
            TextPart(text="private output").mark_as_temp(),
        ],
    )
    runtime.append(reply)
    raw_output = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "public reply"},
            {"type": "text", "text": "private output"},
        ],
    }
    store = bridge.owner.store
    output_id = store.append(state["scope"], raw_output, round_id=state["run"])
    state["output_ids"].append(output_id)
    state["output_messages"].append((output_id, reply))
    store.finish(state["run"], activate=False)

    def broken_sync(*args, **kwargs):
        raise RuntimeError("crash after native commit")

    bridge.sync = broken_sync
    await InternalAgentSubStage._save_to_history(
        SimpleNamespace(conv_manager=bridge.manager),
        event,
        state["request"],
        LLMResponse(role="assistant", completion_text="public reply"),
        runtime,
        None,
    )
    assert state["history_sync_failed"]
    assert store.pending_histories(state["scope"])
    assert "private output" not in json.dumps(
        bridge.manager.histories[("group", "conversation")]
    )
    bridge.close()
    bridge.owner.store = Store(store.path)
    bridge.owner.store.recover()
    restarted = HistoryBridge(bridge.owner)
    try:
        await restarted.refresh("group", "conversation")
        nodes = restarted.owner.store.view(state["scope"])["nodes"]
        text = json.dumps(projected(restarted, state["scope"]))
        assert "public reply" in text
        assert "private output" not in text
        revision = next(node for node in nodes if output_id in node["children"])
        assert revision["active"]
        raw = restarted.owner.store.node(state["scope"], output_id)
        assert raw["payload"]["message"] == raw_output
        assert not raw["active"]
        assert not restarted.owner.store.pending_histories(state["scope"])
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_restart_replays_empty_commit_without_exposing_private_input(bridge):
    current = Message(role="user", content="private current input")
    current._no_save = True
    event, state, runtime, input_id = await registered_round(bridge, current)
    store = bridge.owner.store
    silent_id = store.append(state["scope"], message("unpublished silent tail"))
    store.finish(state["run"], status="cancelled", activate=False)

    def broken_sync(*args, **kwargs):
        raise RuntimeError("crash after empty native commit")

    bridge.sync = broken_sync
    await InternalAgentSubStage._save_to_history(
        SimpleNamespace(conv_manager=bridge.manager),
        event,
        state["request"],
        LLMResponse(role="assistant", completion_text=""),
        runtime,
        None,
        user_aborted=True,
    )
    assert bridge.manager.histories[("group", "conversation")] == []
    assert state["history_sync_failed"]
    assert store.pending_histories(state["scope"])
    bridge.close()
    bridge.owner.store = Store(store.path)
    bridge.owner.store.recover()
    restarted = HistoryBridge(bridge.owner)
    try:
        await restarted.refresh("group", "conversation")
        assert projected(restarted, state["scope"]) == [
            message("unpublished silent tail")
        ]
        assert not restarted.owner.store.node(state["scope"], input_id)["active"]
        assert restarted.owner.store.node(state["scope"], silent_id)["active"]
        assert "private current input" in json.dumps(
            restarted.owner.store.node(state["scope"], input_id)["payload"]
        )
        assert not restarted.owner.store.pending_histories(state["scope"])
    finally:
        restarted.close()
