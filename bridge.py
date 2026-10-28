"""Event-scoped runner integration without changing AstrBot source files."""

from __future__ import annotations

import asyncio
import copy
import inspect
from types import MethodType

from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import ToolSet

from .image_context import ImageContext
from .model_time import model_times


def plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict") and callable(value.dict):
        return plain(value.dict())
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


class ProviderView:
    """Record the actual logical provider input, including fallbacks and retries."""

    def __init__(self, provider, owner, state):
        self._provider, self._owner, self._state = provider, owner, state

    def __getattr__(self, name):
        return getattr(self._provider, name)

    async def _capture(self, kwargs, image_window=None):
        tools = kwargs.get("func_tool")
        schemas = (
            [
                item["function"]
                for item in ToolSet(
                    [tool for tool in tools.tools if tool.active]
                ).openai_schema()
            ]
            if tools
            else []
        )
        data = {
            k: plain(kwargs[k])
            for k in (
                "contexts",
                "prompt",
                "system_prompt",
                "image_urls",
                "audio_urls",
                "extra_user_content_parts",
                "model",
                "session_id",
                "conversation_id",
            )
            if k in kwargs
        }
        data.update(
            tools=copy.deepcopy(schemas),
            provider=self._provider.provider_config.get("id", ""),
            model=kwargs.get("model") or self._provider.get_model(),
            input_kind="provider_logical_request",
            token_method="estimate",
        )
        if image_window is not None:
            data["history_image_window"] = image_window
        limit = self._owner.hard_limit(self._provider)
        if limit > 0 and self._owner.store.count(data) > limit:
            raise ValueError(
                "LCM context exceeds configured model input budget; originals retained, waiting for background compaction"
            )
        event = self._state.get("event")
        if event is not None:
            data["context_sources"] = self._owner.context_rules().manifest(event)
        self._owner.store.snapshot(self._state["scope"], self._state["run"], data)

    async def text_chat(self, **kwargs):
        kwargs, image_window = await self._project(kwargs)
        await self._capture(kwargs, image_window)
        return await self._provider.text_chat(**kwargs)

    async def text_chat_stream(self, **kwargs):
        kwargs, image_window = await self._project(kwargs)
        await self._capture(kwargs, image_window)
        async for response in self._provider.text_chat_stream(**kwargs):
            yield response

    async def _project(self, kwargs):
        contexts = kwargs.get("contexts") or []
        # The prefix contains system + committed history + this round's inputs.
        # Current inputs and every tool continuation live outside this history slice.
        end = max(
            0,
            self._state.get("offset", len(contexts))
            - len(self._state.get("inputs", [])),
        )
        keep = max(0, int(self._owner.config.get("history_image_limit", 5)))
        mode = self._owner.config.get("history_image_mode", "count")
        budget = (
            max(0, int(self._owner.config.get("history_image_token_budget", 4096)))
            if mode == "tokens"
            else None
        )
        projection = ImageContext(self._owner)
        projected = await projection.history(
            contexts[:end],
            keep,
            self._state.get("base", {}).get("nodes", []),
            token_budget=budget,
            model=kwargs.get("model") or self._provider.get_model(),
        )
        values = projected + list(contexts[end:])
        event = self._state.get("event")
        if event is not None:
            from astrbot.core.agent.message import Message

            rules = self._owner.context_rules()
            available = kwargs.get("func_tool")
            call_tools, blocks = rules.provider_tools(
                event, available, self._state.get("tool_catalog", available)
            )
            kwargs = {**kwargs, "func_tool": call_tools}
            if values and plain(values[0]).get("role") == "system":
                value = plain(values[0])
                content = value.get("content", "")
                text = (
                    content
                    if isinstance(content, str)
                    else "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    )
                )
                value["content"] = rules.system(event, text, blocks)
                values[0] = (
                    value
                    if isinstance(values[0], dict)
                    else type(values[0]).model_validate(value)
                )
            else:
                value = {
                    "role": "system",
                    "content": rules.system(
                        event, kwargs.get("system_prompt", ""), blocks
                    ),
                }
                values.insert(
                    0,
                    value
                    if not values or isinstance(values[0], dict)
                    else Message.model_validate(value),
                )
                kwargs = {**kwargs, "system_prompt": ""}
            # Detached input only: this message never enters runner history or
            # native persistence. The request snapshot still records it.
            temporary = {"role": "user", "content": rules.temporary(event, blocks)}
            values.append(
                temporary
                if not values or isinstance(values[0], dict)
                else Message.model_validate(temporary)
            )
        zone = self._owner.time_zone(self._state["scope"])
        values = [
            model_times(value, zone)
            if isinstance(value, dict)
            else type(value).model_validate(model_times(plain(value), zone))
            for value in values
        ]
        return {**kwargs, "contexts": values}, projection.stats


class UnchangedContext:
    """Keep the round's base fixed; never use native dropping/compaction."""

    async def process(self, messages, **kwargs):
        return messages


class RunnerBridge:
    """Patch reset once and scope every effect to a prepared LCM event."""

    def __init__(self, owner):
        if getattr(ToolLoopAgentRunner, "_lcm_bridge", None) is not None:
            raise RuntimeError(
                "LCM runner integration already active; unload previous instance first"
            )
        self.owner = owner
        self.original = ToolLoopAgentRunner.reset
        signature = inspect.signature(self.original)
        bridge = self

        async def reset(runner, *args, **kwargs):
            bound = signature.bind(runner, *args, **kwargs)
            event = getattr(
                getattr(bound.arguments.get("run_context"), "context", None),
                "event",
                None,
            )
            state = event.get_extra("lcm_prepared") if event is not None else None
            tool_catalog = getattr(bound.arguments.get("request"), "func_tool", None)
            await bridge.original(runner, *args, **kwargs)
            if (
                not state
                or state.get("owner") is not bridge.owner
                or bound.arguments.get("request") is not state.get("request")
            ):
                return
            owner = bridge.owner
            state["run"] = owner.store.begin(
                state["scope"], state["reservation"], state["base"], state["inputs"]
            )
            state["offset"] = len(runner.run_context.messages)
            state["outputs"] = 0
            state["runner"] = runner
            state["event"] = event
            # Preserve the full allowed catalog across skills-like subsets and
            # the final no-tools request, so instructions do not change per step.
            state["tool_catalog"] = tool_catalog
            history = getattr(owner, "history_bridge", None)
            if history:
                await history.register(event, state)
            event.set_extra("lcm_run", state["run"])
            runner.request_context_manager = UnchangedContext()
            # Native overflow spilling replaces tool text. LCM retains it in full.
            runner.tool_result_overflow_dir = None
            runner.provider = ProviderView(runner.provider, owner, state)
            runner.fallback_providers = [
                ProviderView(p, owner, state) for p in runner.fallback_providers
            ]
            original_step = runner.step

            def capture():
                new = runner.run_context.messages[state["offset"] :]
                for message in new[state["outputs"] :]:
                    ident = owner.store.append(
                        state["scope"],
                        plain(message),
                        round_id=state["run"],
                        sender="机器人",
                        source=f"{state['run']}:{state['outputs']}",
                    )
                    if history:
                        from .history_bridge import serialized

                        state["output_ids"].append(ident)
                        state["output_messages"].append((ident, message))
                        # Serializing each output separately preserves its exact
                        # native alias; pair validation happens on the full save.
                        for alias in serialized([message]):
                            state["aliases"].append({"id": ident, "message": alias})
                    state["outputs"] += 1
                owner.wake.set()

            async def step(self):
                step_failed = False
                try:
                    if history:
                        await history.rebase(event, state)
                    async for result in original_step():
                        capture()
                        yield result
                except BaseException as exc:
                    step_failed = True
                    capture()
                    owner.store.finish(
                        state["run"],
                        "cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "failed",
                        type(exc).__name__,
                        activate=history is None,
                    )
                    raise
                finally:
                    capture()
                    if runner.done() and not step_failed:
                        final = runner.get_final_llm_resp()
                        status = (
                            "cancelled"
                            if runner.was_aborted()
                            else "complete"
                            if (final and final.role == "assistant")
                            or event.get_extra("silence_completed", False)
                            else "failed"
                        )
                        owner.store.finish(
                            state["run"], status, activate=history is None
                        )

            runner.step = MethodType(step, runner)

        reset.__signature__ = signature
        self.replacement = reset
        ToolLoopAgentRunner.reset = reset
        ToolLoopAgentRunner._lcm_bridge = self

    def close(self):
        if ToolLoopAgentRunner.reset is self.replacement:
            ToolLoopAgentRunner.reset = self.original
        if getattr(ToolLoopAgentRunner, "_lcm_bridge", None) is self:
            delattr(ToolLoopAgentRunner, "_lcm_bridge")
