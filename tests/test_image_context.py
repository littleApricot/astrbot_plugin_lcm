"""Check history-only image limits and original-image retrieval end to end."""

import asyncio
import base64
import copy
import json
from types import SimpleNamespace

import pytest
from astrbot.api.message_components import Image
from astrbot.core.agent.message import Message
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.provider.entities import LLMResponse
from mcp.types import CallToolResult, ImageContent, TextContent

from astrbot_plugin_lcm.bridge import ProviderView, RunnerBridge
from astrbot_plugin_lcm.history_bridge import HistoryBridge
from astrbot_plugin_lcm.image_context import ImageContext
from astrbot_plugin_lcm.main import HistoryTool
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_effective_pipeline import MemoryConversationManager
from astrbot_plugin_lcm.tests.test_integration import FakeProvider, prepare, runner
from astrbot_plugin_meme_library.main import MemeLibrary
from astrbot_plugin_meme_library.store import Library

owner = test_integration.owner
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jQ0kAAAAASUVORK5CYII="
IMAGE = "data:image/png;base64," + PNG


def message(label, count=1, reference=IMAGE):
    return {
        "role": "user",
        "content": [{"type": "text", "text": label}]
        + [
            {"type": "image_url", "image_url": {"url": reference}} for _ in range(count)
        ],
    }


def image_count(messages):
    return sum(
        part.get("type") == "image_url"
        for value in messages
        for part in (
            (value if isinstance(value, dict) else value.model_dump()).get("content")
            or []
        )
        if isinstance(part, dict)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [0, 1, 5, 20])
async def test_oldest_images_become_text_without_changing_originals(owner, keep):
    nodes = []
    for index in range(4):
        ident = owner.store.append(owner.scope, message(str(index), 2))
        nodes.append(owner.store.node(owner.scope, ident))
    originals = owner.messages(nodes)
    before = copy.deepcopy(originals)
    result = await ImageContext(owner).history(originals, keep, nodes)
    assert image_count(result) == min(keep, 8)
    parts = [part for item in result for part in item["content"]]
    projected = [
        part
        for part in parts
        if part.get("type") == "image_url" or part.get("text") == "图片"
    ]
    assert [part["type"] for part in projected] == ["text"] * max(0, 8 - keep) + [
        "image_url"
    ] * min(8, keep)
    assert all(node["id"] in str(result) for node in nodes)
    assert before == originals
    assert image_count(owner.messages(nodes)) == 8


@pytest.mark.asyncio
async def test_existing_gallery_document_is_used_and_reflects_edits(owner, tmp_path):
    from collections import OrderedDict

    library = Library(tmp_path / "gallery")
    try:
        row, _ = library.add(base64.b64decode(PNG), ".png", "group", "", "")
        library.set_status(row["id"], "processing")
        library.recognize(row["id"], "拿捏", "表示被拿捏、求饶。", True)
        plugin = object.__new__(MemeLibrary)
        plugin.config, plugin.library, plugin._decision_images = (
            {},
            library,
            OrderedDict(),
        )
        owner.context = SimpleNamespace(
            get_registered_star=lambda name: SimpleNamespace(star_cls=plugin)
        )
        result = await ImageContext(owner).history([message("old")], 0)
        assert "拿捏" in str(result) and "求饶" in str(result)
        assert "base64" not in str(result)
        current = library.get(row["id"])
        library.edit(
            row["id"], "纠正后的名称", "新的用法", current["revision"], "test", "test"
        )
        result = await ImageContext(owner).history([message("old")], 0)
        assert "纠正后的名称" in str(result) and "新的用法" in str(result)
    finally:
        library.close()


@pytest.mark.asyncio
async def test_summary_worker_sends_text_only_including_serialized_images(owner):
    owner.config["base_tokens"] = 1500
    for index in range(4):
        payload = message("history" + str(index))
        if index == 1:
            payload["content"].append(
                {"type": "text", "text": json.dumps(message("tool"))}
            )
        owner.store.append(owner.scope, payload, tokens=800)
    response = LLMResponse(role="assistant", completion_text="图片相关的中性历史摘要")
    response.raw_completion = {"choices": [{"finish_reason": "stop"}]}
    provider = FakeProvider([response])
    owner.context = SimpleNamespace(
        get_using_provider_async=lambda origin: asyncio.sleep(0, result=provider)
    )
    owner.logger = SimpleNamespace(warning=lambda *a: None)
    task = asyncio.create_task(owner.work())
    try:
        for _ in range(50):
            if any(node["level"] for node in owner.store.view(owner.scope)["nodes"]):
                break
            await asyncio.sleep(0.01)
        assert len(provider.calls) == 1
        assert image_count(provider.calls[0]["contexts"]) == 0
        assert "base64" not in str(provider.calls[0])
        assert "图片" in str(provider.calls[0])
        originals = owner.store.history(owner.scope)["records"]
        assert "data:image/" in str(originals)
        assert any(node["level"] for node in owner.store.view(owner.scope)["nodes"])
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class ViewImages(FunctionTool):
    def __init__(self):
        super().__init__(
            name="view_many_memes",
            description="Return images in this tool round.",
            parameters={"type": "object", "properties": {}},
        )

    async def call(self, context):
        return CallToolResult(
            content=[TextContent(type="text", text="本轮选图")]
            + [
                ImageContent(type="image", mimeType="image/png", data=PNG)
                for _ in range(7)
            ]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("mode", ["count", "tokens"])
async def test_native_tool_loop_preserves_current_and_tool_images(
    owner, shared, mode, monkeypatch, tmp_path
):
    from astrbot.core.agent.tool_image_cache import tool_image_cache

    monkeypatch.setattr(tool_image_cache, "_cache_dir", str(tmp_path / "tool_images"))
    owner.config.update(history_image_mode=mode, history_image_token_budget=920)
    for index in range(8):
        owner.store.append(owner.scope, message("history" + str(index)))
    history = None
    if shared:
        owner.context = SimpleNamespace(
            conversation_manager=MemoryConversationManager()
        )
        history = owner.history_bridge = HistoryBridge(owner)
    execution = RunnerBridge(owner)
    try:
        ev, req = await prepare(owner, "current images")
        req.image_urls = [IMAGE, IMAGE]
        req.func_tool.add_tool(ViewImages())
        provider = FakeProvider(
            [
                LLMResponse(
                    role="assistant",
                    tools_call_name=["view_many_memes"],
                    tools_call_args=[{}],
                    tools_call_ids=["view-1"],
                ),
                LLMResponse(role="assistant", completion_text="finished"),
            ]
        )
        provider.get_model = lambda: "deepseek-flash"
        loop = await runner(owner, ev, req, provider)
        loop.tool_executor = FunctionToolExecutor()
        async for _ in loop.step():
            pass
        assert (
            image_count(provider.calls[0]["contexts"]) == 7
        )  # five history + two current
        if shared:
            owner.store.append(owner.scope, message("new concurrent history"))
        async for _ in loop.step():
            pass
        assert (
            image_count(provider.calls[1]["contexts"]) == 14
        )  # plus seven tool images
        assert image_count(loop.run_context.messages) == 17 + int(
            shared
        )  # raw eight history + two current + seven tool
        if shared:
            old = next(
                item
                for item in provider.calls[1]["contexts"]
                if item.role == "user"
                and item.content
                and item.content[0].model_dump().get("text") == "history3"
            )
            assert old.content[1].model_dump()["type"] == "text"
        snapshots = owner.store.requests(owner.scope, ev.get_extra("lcm_run"))
        latest = owner.store.request(owner.scope, snapshots[-1]["id"])
        assert image_count(latest["payload"]["contexts"]) == 14
        window = latest["payload"]["history_image_window"]
        assert window["mode"] == mode and window["retained_images"] == 5
        if mode == "tokens":
            assert window["estimated_tokens"] == 920
    finally:
        execution.close()
        if history:
            history.close()


@pytest.mark.asyncio
async def test_retrieval_ids_reopen_original_bytes_after_image_projection(
    owner, tmp_path
):
    ident = owner.store.append(owner.scope, message("找回这张图片"))
    path = tmp_path / "original.png"
    original = base64.b64decode(PNG)
    path.write_bytes(original)
    media = await owner.media.capture(owner.scope, ident, Image.fromFileSystem(path))
    path.unlink()
    node = owner.store.node(owner.scope, ident)
    projected = await ImageContext(owner).history(owner.messages([node]), 0, [node])
    assert ident in str(projected) and media in str(projected)
    assert "base64" not in str(projected)
    tool = HistoryTool(owner, owner.scope, "lcm_expand", "test", {}, [])
    record = json.loads(await tool.call(None, id=ident))
    assert "base64" not in str(record)
    discovered = record["units"][0]["records"][0]["media"][0]
    assert discovered["id"] == media and discovered["status"] == "ready"
    assert discovered["path"] and discovered["node"] == ident
    reopened = await tool.call(None, id=media)
    assert isinstance(reopened, CallToolResult)
    assert base64.b64decode(reopened.content[1].data) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["count", "tokens"])
async def test_streaming_uses_same_history_only_limit(owner, mode):
    owner.config.update(history_image_mode=mode, history_image_token_budget=920)
    provider = FakeProvider()
    provider.get_model = lambda: "deepseek-flash"

    async def stream(**kwargs):
        provider.calls.append(kwargs)
        yield LLMResponse(role="assistant", completion_text="stream finished")

    provider.text_chat_stream = stream
    scope = owner.scope
    current = owner.store.append(scope, message("current"))
    base = owner.store.view(scope, exclude=[current])
    owner.store.reserve(scope, [current], "r")
    run = owner.store.begin(scope, "r", base, [current])
    state = {
        "scope": scope,
        "run": run,
        "offset": 8,
        "inputs": [current],
        "base": {"nodes": []},
    }
    view = ProviderView(provider, owner, state)
    contexts = [Message.model_validate(message(str(index))) for index in range(9)]
    async for _ in view.text_chat_stream(contexts=contexts):
        pass
    assert (
        image_count(provider.calls[0]["contexts"]) == 7
    )  # five history + current + tool tail
    assert image_count(contexts) == 9
