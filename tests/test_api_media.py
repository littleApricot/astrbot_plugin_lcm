"""Check authenticated route binding, scope isolation, and retained local bytes."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from astrbot.api.message_components import Image
from astrbot.dashboard.api.plugins import _call_plugin_extension
from fastapi import FastAPI, Request
from mcp.types import CallToolResult, ImageContent

from astrbot_plugin_lcm.main import LCM, HistoryTool


@pytest.mark.asyncio
async def test_read_routes_media_and_cross_session_rejection(tmp_path):
    registered = []
    with patch("astrbot_plugin_lcm.main.StarTools.get_data_dir", return_value=tmp_path):
        obj = LCM(
            SimpleNamespace(
                register_web_api=lambda *a: registered.append(a),
                add_llm_tools=lambda *a: None,
            ),
            {"base_tokens": 40000},
        )
    a = obj.store.scope("groupA", "c")
    b = obj.store.scope("groupB", "c")
    n = obj.store.append(a, {"role": "user", "content": "秘密 Alice"})
    source = tmp_path / "sample.png"
    source.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jQ0kAAAAASUVORK5CYII="
        )
    )
    asset = await obj.media.capture(a, n, Image.fromFileSystem(source))
    assert await obj.media.capture(a, n, Image.fromFileSystem(source)) == asset
    source.unlink()
    search = HistoryTool(obj, a, "lcm_search", "test", {}, [])
    found = json.loads(await search.call(None, query="Alice"))
    assert found["records"][0]["media"][0]["id"] == asset
    expand = HistoryTool(obj, a, "lcm_expand", "test", {}, [])
    result = await expand.call(None, id=asset)
    assert isinstance(result, CallToolResult)
    assert isinstance(result.content[1], ImageContent)
    assert base64.b64decode(result.content[1].data).startswith(b"\x89PNG")
    app = FastAPI()
    app.state.core_lifecycle = SimpleNamespace(
        star_context=SimpleNamespace(registered_web_apis=registered)
    )

    @app.get("/api/{endpoint}")
    async def api(endpoint: str, request: Request):
        return await _call_plugin_extension(
            "astrbot_plugin_lcm/" + endpoint, request, "test"
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/context", params={"scope": a})
        assert response.status_code == 200
        result = response.json()
        assert result["records"][0]["media"][0]["id"] == asset
        response = await client.get("/api/media", params={"scope": a, "id": asset})
        assert response.status_code == 200 and response.json()["data_url"].startswith(
            "data:image/png;base64,"
        )
        for endpoint, ident in (("describe", n), ("expand", n), ("media", asset)):
            response = await client.get(
                "/api/" + endpoint, params={"scope": b, "id": ident}
            )
            assert response.status_code == 400
        response = await client.get("/api/search", params={"scope": a, "q": "alice"})
        assert response.json()["records"][0]["id"] == n
        response = await client.get(
            "/api/context", params={"scope": a, "version": "-1"}
        )
        assert response.status_code == 400
        response = await client.post("/api/context", json={"scope": a})
        assert response.status_code == 405
