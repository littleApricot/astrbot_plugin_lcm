"""Follow declarations into LCM, reply judgment, and the real summary worker."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_integration import FakeProvider, event, prepare
from astrbot_plugin_lcm.tests.test_system_tools import management
from astrbot_plugin_reply_gate.decision_input import build_input, decision_memory
from astrbot_plugin_tool_prompts.context_policy import PLACEMENTS

owner = test_integration.owner


def install(owner):
    counts = {"temporary": 0}

    def candidates(ev):
        counts["temporary"] += 1
        return "REPLY_CANDIDATES"

    declaration = {
        "owner": "content_producer",
        "prompt_blocks": [
            {"key": "memory", "kind": "memory", "content": "RESIDENT_MEMORY"},
            {
                "key": "catalog",
                "kind": "notebook_catalog",
                "content": "NOTEBOOK_CATALOG",
            },
            {"key": "guidance", "kind": "tool_guidance", "content": "TOOL_GUIDANCE"},
            {"key": "candidates", "kind": "meme_candidates", "content": candidates},
        ],
        # New kinds must take precedence over legacy placement declarations.
        "system": "LEGACY_WRONG_POSITION",
        "temporary": "LEGACY_DUPLICATE",
    }
    stars = {
        "astrbot_plugin_lcm": SimpleNamespace(star_cls=owner),
        "producer": SimpleNamespace(
            star_cls=SimpleNamespace(lcm_context_rules=lambda: declaration)
        ),
        "astrbot_plugin_tool_prompts": SimpleNamespace(
            star_cls=management(owner.context)
        ),
    }
    owner.context.get_all_stars = lambda: list(stars.values())
    owner.context.get_registered_star = stars.get
    owner.context.get_config = lambda **kwargs: {"timezone": "Asia/Shanghai"}
    return stars, counts


def test_only_manager_decides_positions_and_legacy_fallback_survives(
    owner, monkeypatch
):
    stars, counts = install(owner)
    ev = event()
    rules = owner.context_rules()
    system = rules.system(ev, "base")
    assert "NOTEBOOK_CATALOG" in system and "TOOL_GUIDANCE" in system
    assert "LEGACY_" not in system and counts["temporary"] == 0
    assert decision_memory(owner.context, ev, "POLLUTED_FALLBACK") == "RESIDENT_MEMORY"
    assert counts["temporary"] == 0
    monkeypatch.setitem(PLACEMENTS, "notebook_catalog", "temporary")
    system = rules.system(ev, system)
    temporary = rules.temporary(ev)
    assert "NOTEBOOK_CATALOG" not in system and "NOTEBOOK_CATALOG" in temporary
    assert "REPLY_CANDIDATES" in temporary and "REPLY_CANDIDATES" not in system
    assert system.count("RESIDENT_MEMORY") == 1
    stars["astrbot_plugin_tool_prompts"].star_cls.running = False
    assert "LEGACY_WRONG_POSITION" in rules.system(ev, system)
    assert rules.provider_tools(ev, "original-tools", "catalog") == (
        "original-tools",
        [],
    )


@pytest.mark.asyncio
async def test_judge_and_summary_exclude_definitions_but_keep_executions(owner):
    _, counts = install(owner)
    source = [
        {"role": "system", "content": "CONTROL_DEFINITIONS"},
        {
            "role": "user",
            "_context_kind": "tool_definitions",
            "content": "CONTROL_INJECTED",
        },
        {
            "role": "user",
            "tools": [{"description": "CONTROL_FIELD"}],
            "content": [
                {"type": "text", "text": "用户原话提到了工具定义，仍应保留。" * 50},
                {
                    "type": "text",
                    "text": "CONTROL_PART",
                    "_context_kind": "notebook_catalog",
                },
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "normal",
                    "type": "function",
                    "function": {"name": "normal_tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "normal", "content": "NORMAL_RESULT" * 50},
        {"role": "assistant", "content": "机器人的实际回答" * 50},
        {"role": "user", "content": "最近的实际消息" * 50},
    ]
    original = copy.deepcopy(source)
    run = owner.store.begin(owner.scope, "source-test", owner.store.view(owner.scope), [])
    for message in source:
        owner.store.append(owner.scope, message, round_id=run)
    owner.store.finish(run)
    owner.store.append(owner.scope, {"role": "user", "content": "保留的新聊天" * 500})
    ev, req = await prepare(owner, "当前消息")
    req.contexts = source
    req.system_prompt = "CONTROL_REQUEST_SYSTEM"
    history, current = await build_input(owner.context, ev, req)
    judge = json.dumps(
        [history, current, decision_memory(owner.context, ev, "CONTROL_BAD_MEMORY")],
        ensure_ascii=False,
    )
    assert "CONTROL_" not in judge
    assert "NORMAL_RESULT" in judge and "normal_tool" in judge
    assert "用户原话提到了工具定义" in judge and "current_time" in judge
    assert "RESIDENT_MEMORY" in judge and counts["temporary"] == 0
    assert req.contexts == original and req.system_prompt == "CONTROL_REQUEST_SYSTEM"
    assert req.func_tool is not None

    response = test_integration.LLMResponse(
        role="assistant", completion_text="有效历史摘要"
    )
    response.raw_completion = {"choices": [{"finish_reason": "stop"}]}
    provider = FakeProvider([response])
    owner.context.get_using_provider_async = lambda origin: asyncio.sleep(
        0, result=provider
    )
    owner.logger = SimpleNamespace(warning=lambda *args: None)
    worker = asyncio.create_task(owner.work())
    try:
        for _ in range(100):
            await asyncio.sleep(0.01)
            if provider.calls:
                break
        assert provider.calls
        material = json.dumps(provider.calls[0]["contexts"], ensure_ascii=False)
        assert "CONTROL_" not in material
        assert "NORMAL_RESULT" in material and "normal_tool" in material
        assert "用户原话提到了工具定义" in material
        assert "TOOL_GUIDANCE" not in material and "NOTEBOOK_CATALOG" not in material
        assert "REPLY_CANDIDATES" not in material and "RESIDENT_MEMORY" not in material
        # The diagnostic archive remains complete and unchanged.
        assert "CONTROL_DEFINITIONS" in str(owner.store.history(owner.scope))
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        ev.cleanup_temporary_local_files()
