"""Follow AstrBot's committed history without coupling LCM to individual tools."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import re
from collections import Counter
from contextvars import ContextVar
from difflib import SequenceMatcher

from astrbot.api import logger
from astrbot.core.agent.message import (
    bind_checkpoint_messages,
    dump_messages_with_checkpoints,
)
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)

from .history_model import history_key, safe_history


def serialized(messages):
    """Use the native serializer, including private message/part save flags."""
    kept, initial_system = [], False
    for message in messages:
        if message.role == "system" and not initial_system:
            initial_system = True
            continue
        if message.role in ("assistant", "user") and message._no_save:
            continue
        kept.append(message)
    return dump_messages_with_checkpoints(kept)


def merge_branch(initial, branch, latest, *, owned=()):
    """Apply a round's edits to unchanged anchors, preserving intervening edits."""
    # Checkpoints remain in AstrBot's native history. They are excluded only
    # from LCM's conversational projection, never from the native write itself.
    initial, branch, latest = map(copy.deepcopy, (initial, branch, latest))

    def keys(values):
        return [history_key(message) for message in values]

    original_keys, latest_keys = keys(initial), keys(latest)
    owned_keys = set(keys(owned))
    anchors = {}
    for block in SequenceMatcher(
        None, original_keys, latest_keys, autojunk=False
    ).get_matching_blocks():
        anchors.update({block.a + i: block.b + i for i in range(block.size)})
    replacements, insertions = {}, {}
    for tag, start, end, nstart, nend in SequenceMatcher(
        None, original_keys, keys(branch), autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        changed = branch[nstart:nend]
        positions = [anchors[i] for i in range(start, end) if i in anchors]
        if start != end:
            # A concurrently replaced/deleted target belongs to the newer writer.
            if len(positions) != end - start:
                # Round-owned additions remain new even when SequenceMatcher
                # combines them with an obsolete prefix replacement.
                fresh = [
                    message for message in changed if history_key(message) in owned_keys
                ]
                insertions.setdefault(len(latest), []).extend(fresh)
                continue
            for position in positions:
                replacements[position] = []
            if positions:
                replacements[positions[0]] = changed
        elif changed:
            if start == len(initial):
                position = len(latest)
            else:
                position = next(
                    (anchors[i] for i in range(start, len(initial)) if i in anchors),
                    len(latest),
                )
            insertions.setdefault(position, []).extend(changed)
    result = []
    for position in range(len(latest) + 1):
        result.extend(insertions.get(position, []))
        if position < len(latest):
            result.extend(replacements.get(position, [latest[position]]))
    # Equal text can anchor to an old message that was concurrently deleted.
    # A round-owned new input must still be present even in that ambiguous diff.
    needed = Counter(keys(owned))
    present = Counter(keys(result))
    missing = []
    for message in owned:
        ident = history_key(message)
        if present[ident] < needed[ident]:
            missing.append(copy.deepcopy(message))
            present[ident] += 1
    if missing:
        position = next(
            (
                i
                for i, message in enumerate(result)
                if history_key(message) in owned_keys
            ),
            len(result),
        )
        result[position:position] = missing
    return result


class HistoryBridge:
    """Observe the shared history API; hold locks only around history writes."""

    def __init__(self, owner):
        self.owner = owner
        self.manager = owner.context.conversation_manager
        existing = getattr(self.manager, "_lcm_history_bridge", None)
        if existing is not None and existing.active:
            raise RuntimeError("LCM history integration already active")
        self.original = self.manager.update_conversation
        self.signature = inspect.signature(self.original)
        self.current = ContextVar("lcm_history_event", default=None)
        self.saving = ContextVar("lcm_native_save", default=None)
        self.locks = {}
        self.pending = {}
        self.active = True
        bridge = self

        async def update(*args, **kwargs):
            if not bridge.active:
                return await bridge.original(*args, **kwargs)
            bound = bridge.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            history = bound.arguments.get("history")
            if history is None:
                return await bridge.original(*args, **kwargs)
            origin = bound.arguments["unified_msg_origin"]
            cid = bound.arguments.get("conversation_id")
            if not cid:
                cid = await bridge.manager.get_curr_conversation_id(origin)
            if not cid:
                return await bridge.original(*args, **kwargs)
            scope = owner.store.scope(origin, str(cid))
            async with bridge.lock(scope):
                before = await bridge.read(origin, cid)
                bridge.replay(scope, before)
                frame = bridge.saving.get()
                event = frame[0] if frame else bridge.current.get()
                state = event.get_extra("lcm_prepared") if event else None
                internal = bool(
                    frame is not None
                    and state
                    and state.get("owner") is owner
                    and state["scope"] == scope
                    and state.get("run")
                    and frame[1] is state.get("request")
                )
                if internal and safe_history(before) != state.get("native_start"):
                    # Parallel reply helpers may already have merged the branch.
                    previous = safe_history(before)
                    incoming = safe_history(history)
                    if not previous or incoming[: len(previous)] != previous:
                        bound.arguments["history"] = merge_branch(
                            state.get("merge_initial", state["initial_history"]),
                            history,
                            before,
                            owned=bridge.owned(state),
                        )
                options = bridge.options(
                    scope,
                    before,
                    bound.arguments["history"],
                    event,
                    state if internal else None,
                )
                prepare = getattr(owner, "saved_history", None)
                if callable(prepare):
                    bound.arguments["history"], aliases = prepare(
                        scope, bound.arguments["history"], options["candidates"]
                    )
                    options["candidates"].extend(aliases)
                if internal:
                    owner.store.stage_history(
                        state["run"], scope, {"before": before, "options": options}
                    )
                try:
                    result = await bridge.original(*bound.args, **bound.kwargs)
                except BaseException:
                    if internal:
                        owner.store.clear_history(state["run"])
                    raise
                try:
                    after = await bridge.read(origin, cid)
                    bridge.sync(
                        scope,
                        before,
                        after,
                        event,
                        state if internal else None,
                        options=options,
                    )
                except Exception:
                    # The native commit succeeded. Repair from the DB before the
                    # next request rather than reporting a misleading save failure.
                    logger.exception("LCM 历史同步失败，下次请求前将重试")
                    if state:
                        state["history_sync_failed"] = True
                return result

        self.replacement = update
        self.manager.update_conversation = update
        self.save_original = InternalAgentSubStage._save_to_history

        async def save(stage, event, *args, **kwargs):
            if not bridge.active:
                return await bridge.save_original(stage, event, *args, **kwargs)
            request = args[0] if args else kwargs.get("req")
            token = bridge.saving.set((event, request))
            try:
                return await bridge.save_original(stage, event, *args, **kwargs)
            finally:
                bridge.saving.reset(token)

        self.save_replacement = save
        InternalAgentSubStage._save_to_history = save
        self.manager._lcm_history_bridge = self

    def lock(self, scope):
        return self.locks.setdefault(scope, asyncio.Lock())

    async def read(self, origin, cid):
        conversation = await self.manager.get_conversation(origin, cid)
        return json.loads(conversation.history or "[]") if conversation else []

    def bind(self, event):
        if event.get_extra("lcm_history_binding") is self:
            return
        token = self.current.set(event)
        cleanup = event.cleanup_temporary_local_files
        event.set_extra("lcm_history_binding", self)

        def done():
            try:
                cleanup()
            finally:
                if self.current.get() is event:
                    try:
                        self.current.reset(token)
                    except ValueError:
                        # Cleanup may run in the adapter's owning task instead.
                        self.current.set(None)

        event.cleanup_temporary_local_files = done

    def candidates(self, scope, event=None, after=()):
        result = []
        ids = [node["id"] for node in self.owner.store.view(scope)["nodes"]]
        state = event.get_extra("lcm_prepared") if event else None
        if state and state.get("scope") == scope:
            ids = list(
                dict.fromkeys(ids + state["inputs"] + state.get("output_ids", []))
            )
        project = getattr(self.owner, "model_nodes", self.owner.store.history_nodes)
        originals = [self.owner.store.node(scope, ident) for ident in ids]
        raw_messages = {
            node["id"]: node["payload"].get("message") for node in originals
        }
        nodes = project(originals)
        # Projection must see the whole tool block, just as the request does.
        # A single-node projection would classify every tool as incomplete.
        for node, message in zip(nodes, self.owner.messages(nodes)):
            ident = node["id"]
            result.append({"id": ident, "message": message})
            if node["level"] == 0:
                raw = raw_messages[ident]
                result.append({"id": ident, "message": raw})
            if node["level"] == 0 and node["payload"].get("extra", {}).get(
                "platform_id"
            ):
                message = copy.deepcopy(node["payload"]["message"])
                content = message.get("content")
                if (
                    isinstance(content, list)
                    and content
                    and content[0].get("type") == "text"
                ):
                    content[0]["text"] = re.sub(
                        r"^\[[^\n]*?\] ", "", content[0]["text"], count=1
                    )
                    result.append({"id": ident, "message": message})
        if event:
            state = event.get_extra("lcm_prepared")
            if state and state.get("scope") == scope:
                result.extend(state.get("aliases", []))
                for ident, message in state.get("output_messages", []):
                    raw = self.owner.store.node(scope, ident)["payload"]["message"]
                    for alias in serialized([message]):
                        result.append(
                            {
                                "id": ident,
                                "message": alias,
                                "project": history_key(raw) != history_key(alias),
                            }
                        )
                original_aliases = {
                    item["id"]: item["message"] for item in state.get("aliases", [])
                }
                for ident, message in state.get("input_messages", []):
                    temporary_parts = isinstance(message.content, list) and any(
                        getattr(part, "_no_save", False) for part in message.content
                    )
                    for alias in serialized([message]):
                        changed = history_key(alias) != history_key(
                            original_aliases.get(ident, alias)
                        )
                        result.append(
                            {
                                "id": ident,
                                "message": alias,
                                "project": temporary_parts or changed,
                            }
                        )
            elif after and event.get_extra("provider_request") is not None:
                # An admission write precedes runner reset. These are the freshly
                # assembled input aliases, including native decorations/media.
                batch = event.get_extra("reply_gate_batch")
                members = (
                    [member.event for member in batch.members] if batch else [event]
                )
                inputs = [member.get_extra("lcm_input") for member in members]
                inputs = [item for item in inputs if item and item["scope"] == scope]
                fresh = [message for message in after if message.get("role") == "user"]
                for item, message in zip(reversed(inputs), reversed(fresh)):
                    result.append({"id": item["id"], "message": message})
        return result

    def options(self, scope, before, after, event=None, state=None):
        fresh = after[len(before) :] if after[: len(before)] == before else []
        return dict(
            candidates=self.candidates(scope, event, fresh),
            run=state.get("run") if state else None,
            internal=state is not None,
            editable=(
                [node["id"] for node in state["base"]["nodes"]]
                + state["inputs"]
                + state.get("output_ids", [])
            )
            if state
            else (),
            excluded=[
                ident
                for ident, message in state.get("input_messages", [])
                if message._no_save
            ]
            if state
            else (),
        )

    def sync(self, scope, before, after, event=None, state=None, *, options=None):
        result = self.owner.store.sync_history(
            scope,
            before,
            after,
            **(
                options
                if options is not None
                else self.options(scope, before, after, event, state)
            ),
        )
        if state:
            self.owner.store.clear_history(state["run"])
            state["history_committed"] = True
            self.pending.pop(state["run"], None)
        self.owner.wake.set()
        return result

    def replay(self, scope, actual):
        for intent in self.owner.store.pending_histories(scope):
            payload = intent["payload"]
            mirrored = self.owner.store.native_history(scope)
            before = mirrored["messages"] if mirrored else payload["before"]
            self.owner.store.sync_history(scope, before, actual, **payload["options"])
            self.owner.store.clear_history(intent["run"])
            pending = self.pending.pop(intent["run"], None)
            if pending:
                pending[1]["history_committed"] = True
            self.owner.wake.set()

    def owned(self, state):
        result = []
        for ident, message in state.get("input_messages", []):
            if ident in state.get("owned_input_ids", []):
                result.extend(serialized([message]))
        for _, message in state.get("output_messages", []):
            result.extend(serialized([message]))
        return result

    async def refresh(self, origin, cid, event=None):
        scope = self.owner.store.scope(origin, str(cid))
        async with self.lock(scope):
            actual = await self.read(origin, cid)
            self.replay(scope, actual)
            previous = self.owner.store.native_history(scope)
            before = previous["messages"] if previous else actual
            for pending_event, state in list(self.pending.values()):
                if state["scope"] == scope and state.get("history_sync_failed"):
                    self.sync(scope, before, actual, pending_event, state)
                    previous = self.owner.store.native_history(scope)
                    before = previous["messages"]
            if previous is None or safe_history(actual) != before:
                self.sync(scope, before, actual, event)
            return safe_history(actual)

    async def refresh_request(self, event, req):
        state = event.get_extra("lcm_prepared")
        if not state or state.get("owner") is not self.owner:
            return
        await self.refresh(state["origin"], state["cid"], event)
        if not state.get("run"):
            state["inputs"] = self.owner.store.reserve(
                state["scope"], state["inputs"], state["reservation"]
            )
        base = getattr(self.owner, "view", self.owner.store.view)(
            state["scope"], exclude=state["inputs"]
        )
        state["base"] = base
        req.contexts = self.owner.messages(base["nodes"])
        batch = event.get_extra("reply_gate_batch")
        if batch is not None:
            req.contexts += copy.deepcopy(batch.past_messages)

    async def rebase(self, event, state):
        """Use the latest effective base without publishing private tool records."""
        await self.refresh(state["origin"], state["cid"], event)
        runner = state["runner"]
        old = runner.run_context.messages
        tail = old[state["offset"] :]
        excluded = [*state["inputs"], *state["output_ids"]]
        base = getattr(self.owner, "view", self.owner.store.view)(
            state["scope"], exclude=excluded
        )
        systems = old[:1] if old and old[0].role == "system" else []
        inputs = [value for _, value in reversed(state["input_messages"])]
        prefix = (
            systems
            + bind_checkpoint_messages(self.owner.messages(base["nodes"]))
            + inputs
        )
        runner.run_context.messages = prefix + tail
        state["offset"] = len(prefix)
        state["base"] = base

    async def register(self, event, state):
        state["initial_history"] = serialized(state["runner"].run_context.messages)
        state["native_start"] = await self.read(state["origin"], state["cid"])
        state["native_start"] = safe_history(state["native_start"])
        state["aliases"] = []
        state["input_messages"] = []
        runtime_users = [
            message
            for message in state["runner"].run_context.messages
            if message.role == "user"
        ]
        for ident, message in zip(reversed(state["inputs"]), reversed(runtime_users)):
            state["input_messages"].append((ident, message))
        # Inputs not yet admitted to native history are additions owned by this
        # round, so a concurrent edit must not silently discard them.
        state["merge_initial"] = copy.deepcopy(state["initial_history"])
        mirrored = self.owner.store.native_history(state["scope"])
        admitted = set(mirrored["ids"]) if mirrored else set()
        state["owned_input_ids"] = [
            ident for ident in state["inputs"] if ident not in admitted
        ]
        for ident, message in state["input_messages"]:
            if ident in admitted:
                continue
            for alias in serialized([message]):
                key = history_key(alias)
                for index in range(len(state["merge_initial"]) - 1, -1, -1):
                    if history_key(state["merge_initial"][index]) == key:
                        state["merge_initial"].pop(index)
                        break
        users = [m for m in state["initial_history"] if m.get("role") == "user"]
        for ident, message in zip(reversed(state["inputs"]), reversed(users)):
            state["aliases"].append({"id": ident, "message": message})
        state["output_ids"] = []
        state["output_messages"] = []
        state["history_committed"] = False
        self.pending[state["run"]] = (event, state)

    def abandon(self, event, state):
        if (
            not state.get("run")
            or state.get("history_committed")
            or state.get("history_sync_failed")
        ):
            return
        previous = self.owner.store.native_history(state["scope"])
        messages = previous["messages"] if previous else []
        self.sync(state["scope"], messages, messages, event, state)

    def close(self):
        self.active = False
        if self.manager.update_conversation is self.replacement:
            self.manager.update_conversation = self.original
        if InternalAgentSubStage._save_to_history is self.save_replacement:
            InternalAgentSubStage._save_to_history = self.save_original
        if getattr(self.manager, "_lcm_history_bridge", None) is self:
            delattr(self.manager, "_lcm_history_bridge")
        self.pending.clear()
