"""Native registration, per-event isolation and editable declaration integration."""

import asyncio
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.provider.func_tool_manager import FunctionToolManager
from astrbot.core.star.context import Context
from astrbot.core.star.star import StarMetadata, star_map
from astrbot.dashboard.services.tools_service import ToolsService

from astrbot_plugin_lcm.bridge import ProviderView
from astrbot_plugin_lcm.main import LCM


@pytest.fixture
def registered(tmp_path, monkeypatch):
    from astrbot.core.provider import func_tool_manager

    monkeypatch.setattr(func_tool_manager.sp, "global_get", AsyncMock(return_value={}))
    module = LCM.__module__
    monkeypatch.setitem(
        star_map,
        module,
        StarMetadata(
            name="astrbot_plugin_lcm",
            display_name="LCM 上下文与历史",
            module_path=module,
        ),
    )
    manager = object.__new__(FunctionToolManager)
    manager.func_list = []
    manager.builtin_func_list = {}
    context = SimpleNamespace(
        provider_manager=SimpleNamespace(llm_tools=manager),
        get_config=lambda **kwargs: {"timezone": "Asia/Shanghai"},
        register_web_api=lambda *args: None,
    )
    context.add_llm_tools = partial(Context.add_llm_tools, context)
    context.get_llm_tool_manager = partial(Context.get_llm_tool_manager, context)
    with patch("astrbot_plugin_lcm.main.StarTools.get_data_dir", return_value=tmp_path):
        owner = LCM(context, {"enabled": True})
    return owner, manager


def event(owner, scope, origin):
    extras = {"lcm_prepared": {"owner": owner, "scope": scope, "origin": origin}}
    return SimpleNamespace(
        unified_msg_origin=origin,
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
        is_admin=lambda: False,
        get_sender_id=lambda: "isolated-user",
        cleanup_temporary_local_files=lambda: None,
    )


async def execute(tool, ev, **kwargs):
    context = SimpleNamespace(context=SimpleNamespace(event=ev), tool_call_timeout=10)
    result = [
        item async for item in FunctionToolExecutor.execute(tool, context, **kwargs)
    ]
    return json.loads(result[0].content[0].text)


@pytest.mark.asyncio
async def test_tools_appear_in_native_list_before_first_message(registered):
    owner, manager = registered
    service = object.__new__(ToolsService)
    service.tool_mgr = manager
    service._get_config_entries = lambda: []
    tools = {item["name"]: item for item in await service.get_tool_list()}
    assert {"lcm_search", "lcm_describe", "lcm_expand"} <= tools.keys()
    for name in ("lcm_search", "lcm_describe", "lcm_expand"):
        assert tools[name]["origin"] == "plugin"
        assert tools[name]["origin_name"] == "astrbot_plugin_lcm"
        assert manager.get_func(name).scope is None
        assert "scope" not in tools[name]["parameters"]["properties"]
    assert tools["lcm_search"]["parameters"]["properties"]["limit"]["type"] == "integer"
    assert tools["lcm_search"]["parameters"]["required"] == ["query"]
    await owner.terminate()
    assert manager.func_list == []


@pytest.mark.asyncio
async def test_same_registered_handler_keeps_parallel_groups_separate(registered):
    owner, manager = registered
    a = owner.store.scope("group-a", "conversation-a")
    b = owner.store.scope("group-b", "conversation-b")
    owner.store.append(a, {"role": "user", "content": "共享关键词：A 的私有内容"})
    owner.store.append(b, {"role": "user", "content": "共享关键词：B 的私有内容"})
    search = manager.get_full_tool_set().get_tool("lcm_search")
    first, second = await asyncio.gather(
        execute(search, event(owner, a, "group-a"), query="共享关键词"),
        execute(search, event(owner, b, "group-b"), query="共享关键词"),
    )
    assert first["ok"] and second["ok"]
    assert "A 的私有内容" in json.dumps(first, ensure_ascii=False)
    assert "B 的私有内容" not in json.dumps(first, ensure_ascii=False)
    assert "B 的私有内容" in json.dumps(second, ensure_ascii=False)
    assert "A 的私有内容" not in json.dumps(second, ensure_ascii=False)
    assert manager.get_func("lcm_search").scope is None
    assert (await execute(search, event(owner, b, "group-a"), query="共享关键词"))[
        "error"
    ] == "lcm_context_unavailable"
    assert not (
        await execute(search, event(owner, a, "group-a"), query="共享关键词", scope=b)
    )["ok"]


@pytest.mark.asyncio
async def test_native_permission_guard_still_applies(registered, monkeypatch):
    from astrbot.core.provider import func_tool_manager

    owner, manager = registered
    monkeypatch.setattr(
        func_tool_manager.sp,
        "global_get",
        AsyncMock(return_value={"_default": {"lcm_search": "admin"}}),
    )
    scope = owner.store.scope("group-a", "c")
    tools = manager.get_full_tool_set()
    context = SimpleNamespace(
        context=SimpleNamespace(event=event(owner, scope, "group-a")),
        tool_call_timeout=10,
    )
    result = [
        item
        async for item in FunctionToolExecutor.execute(
            tools.get_tool("lcm_search"), context, query="x"
        )
    ]
    assert "Permission denied" in result[0].content[0].text


@pytest.mark.asyncio
async def test_prepare_does_not_restore_disabled_tools(registered):
    owner, manager = registered
    scope = owner.store.scope("group-a", "c")
    mid = owner.store.append(scope, {"role": "user", "content": "new input"})
    ev = event(owner, scope, "group-a")
    ev.set_extra("lcm_prepared", None)
    ev.set_extra("lcm_input", {"scope": scope, "id": mid})
    manager.get_func("lcm_search").active = False
    req = ProviderRequest(
        prompt="new input",
        conversation=SimpleNamespace(cid="c"),
        func_tool=ToolSet(
            [tool for tool in manager.get_full_tool_set().tools if tool.active]
        ),
    )
    await owner.prepare(ev, req)
    assert {tool.name for tool in req.func_tool.tools} == {"lcm_describe", "lcm_expand"}
    ev.cleanup_temporary_local_files()


@pytest.mark.asyncio
async def test_reload_cleanup_does_not_remove_new_instances_tools(registered, tmp_path):
    old, manager = registered
    with patch("astrbot_plugin_lcm.main.StarTools.get_data_dir", return_value=tmp_path):
        new = LCM(old.context, {"enabled": True})
    await old.terminate()
    assert {tool.name for tool in manager.func_list} == {
        "lcm_search",
        "lcm_describe",
        "lcm_expand",
    }
    assert all(tool.owner is new and tool.active for tool in manager.func_list)
    await new.terminate()
    assert manager.func_list == []


@pytest.mark.asyncio
async def test_editor_changes_match_lcm_snapshot_and_budget(registered, tmp_path):
    descriptions = pytest.importorskip("astrbot_plugin_tool_prompts.descriptions")
    integration = pytest.importorskip("astrbot_plugin_tool_prompts.schema_bridge")
    owner, manager = registered
    store = descriptions.DescriptionStore(tmp_path / "tool-overrides.json")
    catalog = descriptions.ToolCatalog(manager, store)
    bridge = integration.SchemaBridge(store)
    bridge.start()
    try:
        item = next(
            item for item in catalog.view()["tools"] if item["name"] == "lcm_search"
        )
        custom = "需要核对原话时主动搜索。" * 300
        catalog.update(
            {
                "id": item["id"],
                "revision": 0,
                "definition_digest": item["definition_digest"],
                "description": custom,
                "parameters": {"/properties/query": "结合当前问题选择关键词"},
            }
        )
        scope = owner.store.scope("group-a", "c")
        run = owner.store.begin(scope, "test", owner.store.view(scope), [])
        provider = SimpleNamespace(
            provider_config={"id": "isolated", "max_context_tokens": 1000000},
            get_model=lambda: "isolated",
        )
        view = ProviderView(provider, owner, {"scope": scope, "run": run})
        tools = manager.get_full_tool_set()
        await view._capture({"prompt": "isolated prompt", "func_tool": tools})
        snapshot = owner.store.request(scope, owner.store.requests(scope, run)[0]["id"])
        actual = next(
            item
            for item in snapshot["payload"]["tools"]
            if item["name"] == "lcm_search"
        )
        assert actual["description"] == custom
        assert (
            actual["parameters"]["properties"]["query"]["description"]
            == "结合当前问题选择关键词"
        )
        assert "description" not in actual["parameters"]["properties"]["sender"]
        assert actual == next(
            item["function"]
            for item in tools.openai_schema()
            if item["function"]["name"] == "lcm_search"
        )
        owner.config["max_input_tokens"] = 100
        with pytest.raises(ValueError, match="input budget"):
            await view._capture({"prompt": "isolated prompt", "func_tool": tools})
        catalog.update(
            {
                "id": item["id"],
                "revision": 1,
                "definition_digest": item["definition_digest"],
            },
            reset=True,
        )
        assert (
            tools.openai_schema()[-1]["function"]["description"]
            == manager.get_func("lcm_search").description
        )
    finally:
        bridge.stop()
