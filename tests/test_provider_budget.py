"""Verify shared model configuration against an actual native provider request."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from astrbot.api import AstrBotConfig
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from openai import AsyncOpenAI

from astrbot_plugin_lcm.bridge import ProviderView
from astrbot_plugin_lcm.main import LCM
from astrbot_plugin_lcm.store import Store


@pytest.mark.parametrize(
    "kind,body,reserve",
    [
        ("openai_chat_completion", {"max_tokens": 16384}, 16384),
        ("openai_chat_completion", {"max_completion_tokens": 32768}, 32768),
        ("openai_responses", {"max_output_tokens": 4096, "max_tokens": 16384}, 4096),
        ("googlegenai_chat_completion", {"maxOutputTokens": "8192"}, 8192),
        ("anthropic_chat_completion", {"max_tokens": 24576}, 24576),
        ("anthropic_chat_completion", None, 65536),
        ("openai_chat_completion", None, 8192),
        ("openai_chat_completion", {"max_tokens": 0}, 8192),
        ("openai_chat_completion", {"max_tokens": ""}, 8192),
    ],
)
def test_read_model_output_configuration(kind, body, reserve):
    owner = object.__new__(LCM)
    owner.config = {"output_reserve_tokens": 1}
    provider = SimpleNamespace(
        provider_config={
            "type": kind,
            "max_context_tokens": 131072,
            "custom_extra_body": body,
        }
    )
    before = json.dumps(provider.provider_config)
    assert owner.hard_limit(provider) == 131072 - reserve
    assert json.dumps(provider.provider_config) == before
    owner.config["max_input_tokens"] = 1000
    assert owner.hard_limit(provider) == 1000


def test_model_switch_and_live_changes_recompute_budget():
    owner = object.__new__(LCM)
    owner.config = {}
    reply = SimpleNamespace(
        provider_config={
            "max_context_tokens": 131072,
            "custom_extra_body": {"max_tokens": 32768},
        }
    )
    summary = SimpleNamespace(
        provider_config={
            "max_context_tokens": 65536,
            "custom_extra_body": {"max_tokens": 8192},
        }
    )
    assert owner.hard_limit(reply) == 98304
    assert owner.hard_limit(summary) == 57344
    reply.provider_config["custom_extra_body"]["max_tokens"] = 16384
    assert owner.hard_limit(reply) == 114688


def test_old_duplicate_option_is_removed_from_saved_config(tmp_path):
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text()
    )
    assert "output_reserve_tokens" not in schema
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"base_tokens": 50000, "output_reserve_tokens": 4096}))
    config = AstrBotConfig(str(path), schema=schema)
    with patch("astrbot_plugin_lcm.main.StarTools.get_data_dir", return_value=tmp_path):
        LCM(SimpleNamespace(register_web_api=lambda *a: None), config)
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    assert (
        "output_reserve_tokens" not in saved and "output_reserve_tokens" not in config
    )
    assert saved["base_tokens"] == 50000


@pytest.mark.parametrize("mode", [None, "tokens"])
def test_image_mode_upgrade_preserves_existing_configuration(tmp_path, mode):
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text()
    )
    settings = {"history_image_limit": 9}
    if mode:
        settings.update(history_image_mode=mode, history_image_token_budget=1536)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings))
    config = AstrBotConfig(str(path), schema=schema)
    assert config["history_image_mode"] == (mode or "count")
    assert config["history_image_token_budget"] == (1536 if mode else 4096)
    assert config["history_image_limit"] == 9
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    assert saved["history_image_mode"] == config["history_image_mode"]


@pytest.mark.asyncio
async def test_native_openai_payload_matches_lcm_reserve_and_budget_guard(tmp_path):
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "isolated",
                "object": "chat.completion",
                "created": 0,
                "model": "isolated",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "test reply"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = ProviderOpenAIOfficial(
        {
            "id": "isolated",
            "type": "openai_chat_completion",
            "key": ["isolated-test-marker"],
            "model": "isolated",
            "api_base": "https://isolated.invalid/v1",
            "max_context_tokens": 16384,
            "custom_extra_body": {"max_tokens": 4096},
        },
        {},
    )
    await provider.client.close()
    provider.client = AsyncOpenAI(
        api_key="isolated-test-marker",
        base_url="https://isolated.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    owner = object.__new__(LCM)
    owner.config = {}
    owner.store = Store(tmp_path / "history.sqlite3")
    scope = owner.store.scope("group", "conv")
    mid = owner.store.append(scope, {"role": "user", "content": "hello"})
    base = owner.store.view(scope, exclude=[mid])
    owner.store.reserve(scope, [mid], "isolated")
    run = owner.store.begin(scope, "isolated", base, [mid])
    wrapped = ProviderView(provider, owner, {"scope": scope, "run": run})
    try:
        assert owner.hard_limit(provider) == 12288
        await wrapped.text_chat(prompt="hello")
        assert calls[0]["max_tokens"] == 4096
        provider.provider_config["custom_extra_body"]["max_tokens"] = 8192
        assert owner.hard_limit(provider) == 8192
        await wrapped.text_chat(prompt="hello again")
        assert calls[1]["max_tokens"] == 8192
        with pytest.raises(ValueError, match="input budget"):
            await wrapped.text_chat(prompt="超出预算" * 10000)
        assert len(calls) == 2
    finally:
        await provider.client.close()
