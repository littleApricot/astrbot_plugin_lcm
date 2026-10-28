"""Check system documentation through real serialization and tool execution."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.register import llm_tools
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from openai import AsyncOpenAI

from astrbot_plugin_lcm.bridge import ProviderView, RunnerBridge, plain
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_integration import FakeProvider, prepare, runner
from astrbot_plugin_lcm.tool_instructions import tool_view
from astrbot_plugin_meme_library.main import RECOMMENDATION_NOTE, MemeLibrary
from astrbot_plugin_reply_gate.decision_input import build_input
from astrbot_plugin_self_prompt.main import MEMORY_REMINDER
from astrbot_plugin_self_prompt.tests.support import Sandbox
from astrbot_plugin_tool_prompts.descriptions import DescriptionStore, tool_id
from astrbot_plugin_tool_prompts.schema_bridge import SchemaBridge

owner = test_integration.owner


def tool(name="a_tool"):
    return FunctionTool(
        name=name,
        description="完整工具用途",
        parameters={
            "type": "object",
            "description": "整体参数说明",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "名称恰好是 description",
                },
                "count": {"type": "integer", "minimum": 1, "default": 2},
                "data": {"$ref": "#/$defs/value"},
            },
            "$defs": {
                "value": {
                    "type": "object",
                    "description": "嵌套说明",
                    "default": {"description": "这是数据"},
                    "properties": {
                        "text": {"type": "string", "description": "文本说明"}
                    },
                }
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    )


def test_stable_catalog_preserves_constraints_and_editor_changes(tmp_path):
    a, z, disabled = tool(), tool("z_tool"), tool("disabled")
    disabled.active = False
    original = copy.deepcopy(a.parameters)
    store = DescriptionStore(tmp_path / "overrides.json")
    bridge = SchemaBridge(store)
    bridge.start()
    try:
        store.save(
            tool_id(a),
            {
                "description": "我的完整用法",
                "parameters": {"/$defs/value/properties/text": "我的嵌套说明"},
            },
            0,
        )
        catalog = ToolSet([z, disabled, a])
        instructions, calls = tool_view(catalog, catalog)
        reordered = ToolSet([a, z, disabled])
        assert tool_view(reordered, reordered)[0] == instructions
        assert instructions.index("## a_tool") < instructions.index("## z_tool")
        assert "disabled" not in instructions
        assert "我的完整用法" in instructions and "我的嵌套说明" in instructions
        assert tool_view(catalog.get_light_tool_set(), catalog)[0] == instructions
        assert (
            tool_view(ToolSet([catalog.get_param_only_tool_set().tools[0]]), catalog)[0]
            == instructions
        )
        assert tool_view(None, catalog)[0] == instructions
        definitions = calls.openai_schema()
        a_call = definitions[0]["function"]
        assert "description" not in a_call
        schema = a_call["parameters"]
        assert "description" in schema["properties"]
        assert "description" not in schema["properties"]["description"]
        assert schema["required"] == ["description"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["data"]["$ref"] == "#/$defs/value"
        assert schema["properties"]["count"] == {
            "type": "integer",
            "minimum": 1,
            "default": 2,
        }
        assert schema["$defs"]["value"]["default"] == {"description": "这是数据"}
        assert "我的完整用法" not in str(calls.anthropic_schema())
        assert "我的完整用法" not in str(calls.google_schema())
        assert a.parameters == original and a.description == "完整工具用途"
        store.save(tool_id(a), {"description": "保存后的新用法"}, 1)
        assert "保存后的新用法" in tool_view(catalog, catalog)[0]
        assert "我的完整用法" not in tool_view(catalog, catalog)[0]
    finally:
        bridge.stop()


@pytest.mark.asyncio
async def test_tool_loop_refreshes_catalog_without_changing_manual_or_judge(owner):
    sandbox = await Sandbox().open()
    execution = RunnerBridge(owner)
    try:
        owner.context = sandbox.context
        gallery = object.__new__(MemeLibrary)
        gallery.config = {}
        stars = {
            "astrbot_plugin_self_prompt": SimpleNamespace(star_cls=sandbox.plugin),
            "astrbot_plugin_lcm": SimpleNamespace(star_cls=owner),
            "astrbot_plugin_meme_library": SimpleNamespace(star_cls=gallery),
        }
        owner.context.get_all_stars = lambda: list(stars.values())
        owner.context.get_registered_star = stars.get
        ev = sandbox.message(2)
        ev.cleanup_temporary_local_files = lambda: None
        ev.trace = SimpleNamespace(record=lambda *a, **k: None)
        ev.get_result = lambda: None
        ev.set_extra("meme_library_temporary", "本轮可选表情推荐：候选甲")
        ev.set_extra("self_prompt_memory_reminder", MEMORY_REMINDER)
        req = await sandbox.prepare(ev)
        req.conversation.cid = "conv"
        req.conversation.token_usage = 0
        req.func_tool = ToolSet(
            [
                copy.copy(llm_tools.get_func(name))
                for name in (
                    "notebook_create",
                    "get_self_prompt",
                )
            ]
        )
        for function in req.func_tool.tools:
            function.handler = getattr(sandbox.plugin, function.name)
        owner.scope = owner.store.scope(ev.unified_msg_origin, "conv")
        owner.store.import_history(owner.scope, [])
        owner.store.append(
            owner.scope, {"role": "user", "content": "较早的完整消息" * 1000}
        )
        mid = owner.store.append(owner.scope, {"role": "user", "content": req.prompt})
        ev.set_extra("lcm_input", {"scope": owner.scope, "id": mid})
        await owner.prepare(ev, req)
        await owner.remove_duplicate_context(ev, req)
        prior_context = copy.deepcopy(req.contexts)
        history, current = await build_input(owner.context, ev, req)
        judge = json.dumps(
            [ev.get_extra("reply_gate_memory"), history, current], ensure_ascii=False
        )
        assert "性格和人设" in judge and "较早的完整消息" in judge
        assert "current_time" in judge
        assert "[记事本目录]" not in judge and "工具使用说明" not in judge
        assert "候选甲" not in judge and MEMORY_REMINDER not in judge

        class Provider(FakeProvider):
            async def text_chat(self, **kwargs):
                response = await super().text_chat(**kwargs)
                self.calls[-1]["tools"] = kwargs["func_tool"].openai_schema()
                return response

        provider = Provider(
            [
                LLMResponse(
                    role="assistant",
                    tools_call_name=["notebook_create"],
                    tools_call_args=[
                        {
                            "title": "新增目录项",
                            "description": "新描述",
                            "content": "仅正文可见内容",
                        }
                    ],
                    tools_call_ids=["create"],
                ),
                LLMResponse(
                    role="assistant",
                    tools_call_name=["get_self_prompt"],
                    tools_call_args=[{}],
                    tools_call_ids=["read"],
                ),
                LLMResponse(role="assistant", completion_text="正常回复"),
            ]
        )
        loop = await runner(owner, ev, req, provider)
        loop.tool_executor = FunctionToolExecutor()
        while not loop.done():
            async for _ in loop.step():
                pass
        assert len(provider.calls) == 3
        systems = [plain(call["contexts"][0])["content"] for call in provider.calls]
        assert "新增目录项" not in systems[0]
        assert "新增目录项" in systems[1] and "新描述" in systems[1]
        assert systems[1] == systems[2]
        manuals = [
            s.split("[系统提示｜来源：astrbot.tools]\n")[1].split("[/系统提示]")[0]
            for s in systems
        ]
        assert manuals[0] == manuals[1] == manuals[2]
        for system, call in zip(systems, provider.calls):
            assert system.count("[记事本目录]") == 1
            assert system.count("[系统提示｜来源：astrbot.tools]") == 1
            assert "仅正文可见内容" not in system
            assert RECOMMENDATION_NOTE in system
            last = plain(call["contexts"][-1])["content"]
            assert "候选甲" in last and MEMORY_REMINDER in last
            assert "[记事本目录]" not in last and RECOMMENDATION_NOTE not in last
            assert "较早的完整消息" in str(call["contexts"])
            assert all("description" not in row["function"] for row in call["tools"])
        assert req.contexts == prior_context
        archived = owner.store.expand(owner.scope, ev.get_extra("lcm_run"))
        assert "工具使用说明" not in str(archived) and "[记事本目录]" not in str(
            archived
        )
    finally:
        execution.close()
        await sandbox.close()


@pytest.mark.asyncio
async def test_native_openai_payload_and_snapshot_agree(owner, tmp_path):
    ev, req = await prepare(owner)
    store = DescriptionStore(tmp_path / "overrides.json")
    bridge = SchemaBridge(store)
    bridge.start()
    source = tool()
    store.save(tool_id(source), {"description": "已编辑的完整用法"}, 0)
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "isolated",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "offline"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = ProviderOpenAIOfficial(
        {
            "id": "isolated",
            "type": "openai_chat_completion",
            "key": ["offline-marker"],
            "model": "isolated",
            "api_base": "https://isolated.invalid/v1",
        },
        {},
    )
    await provider.client.close()
    provider.client = AsyncOpenAI(
        api_key="offline-marker",
        base_url="https://isolated.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    state = ev.get_extra("lcm_prepared")
    state["run"] = owner.store.begin(
        owner.scope, state["reservation"], state["base"], []
    )
    state["event"] = ev
    state["offset"] = 1
    try:
        view = ProviderView(provider, owner, state)
        await view.text_chat(
            contexts=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": "当前消息"},
            ],
            func_tool=ToolSet([source]),
        )
        payload = calls[0]
        assert payload["messages"][0]["content"].count("已编辑的完整用法") == 1
        definition = payload["tools"][0]["function"]
        assert "description" not in definition
        assert definition["parameters"]["required"] == ["description"]
        snapshot = owner.store.request(
            owner.scope, owner.store.requests(owner.scope, state["run"])[0]["id"]
        )
        assert snapshot["payload"]["tools"] == [definition]
        assert (
            snapshot["payload"]["contexts"][0]["content"]
            == payload["messages"][0]["content"]
        )
    finally:
        await provider.client.close()
        bridge.stop()
