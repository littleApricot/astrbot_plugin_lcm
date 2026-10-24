"""Event-scoped runner integration without changing AstrBot source files."""

from __future__ import annotations

import asyncio
import copy
import inspect
from types import MethodType

from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner


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

    async def _capture(self, kwargs):
        tools = kwargs.get("func_tool")
        schemas = (
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools.tools
                if t.active
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
        limit = self._owner.hard_limit(self._provider)
        if limit > 0 and self._owner.store.count(data) > limit:
            raise ValueError(
                "LCM context exceeds configured model input budget; originals retained, waiting for background compaction"
            )
        self._owner.store.snapshot(self._state["scope"], self._state["run"], data)

    async def text_chat(self, **kwargs):
        await self._capture(kwargs)
        return await self._provider.text_chat(**kwargs)

    async def text_chat_stream(self, **kwargs):
        await self._capture(kwargs)
        async for response in self._provider.text_chat_stream(**kwargs):
            yield response


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
                    owner.store.append(
                        state["scope"],
                        plain(message),
                        round_id=state["run"],
                        sender="机器人",
                        source=f"{state['run']}:{state['outputs']}",
                    )
                    state["outputs"] += 1
                owner.wake.set()

            async def step(self):
                try:
                    async for result in original_step():
                        capture()
                        yield result
                except BaseException as exc:
                    capture()
                    owner.store.finish(
                        state["run"],
                        "cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "failed",
                        type(exc).__name__,
                    )
                    raise
                finally:
                    capture()
                    if runner.done():
                        final = runner.get_final_llm_resp()
                        status = (
                            "cancelled"
                            if runner.was_aborted()
                            else "complete"
                            if final and final.role == "assistant"
                            else "failed"
                        )
                        owner.store.finish(state["run"], status)

            runner.step = MethodType(step, runner)

        self.replacement = reset
        ToolLoopAgentRunner.reset = reset
        ToolLoopAgentRunner._lcm_bridge = self

    def close(self):
        if ToolLoopAgentRunner.reset is self.replacement:
            ToolLoopAgentRunner.reset = self.original
        if getattr(ToolLoopAgentRunner, "_lcm_bridge", None) is self:
            delattr(ToolLoopAgentRunner, "_lcm_bridge")
