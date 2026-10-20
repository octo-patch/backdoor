"""Exercise setup, saved settings, and serialized requests without provider calls."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from proxy import routes
from proxy.client import ProviderClient
from proxy.config import Settings, get_settings


ROOT = Path(__file__).parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wizard(monkeypatch, tmp_path):
    module = load_module("wizard")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "wizard.py"))
    for name in ("animate_backdoor", "torus_to_genie", "cls", "_show_genie", "w"):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "ask_power_tools", lambda: [])
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module.os, "execv", lambda *args: None)
    return module


def setup(wizard, monkeypatch, tmp_path, provider_name, model):
    index = next(i for i, p in enumerate(wizard.PROVIDERS, 1) if p["name"] == provider_name)
    answers = ["Tester", str(index), model]
    provider = wizard.PROVIDERS[index - 1]
    if provider["key_url"]:
        answers.append("offline-test-key")
    if provider_name == "Custom":
        answers.append("https://custom.example/v1")
    answers.append("")
    inputs = iter(answers)
    monkeypatch.setattr("builtins.input", lambda: next(inputs))
    wizard.run_wizard()
    return Settings(_env_file=tmp_path / ".env")


@pytest.mark.parametrize("provider_name,host", [
    ("MiniMax", "api.minimax.io"),
    ("MiniMax (China)", "api.minimaxi.com"),
])
@pytest.mark.parametrize("model", ["", "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
@pytest.mark.parametrize("stream", [False, True])
async def test_setup_to_http(wizard, monkeypatch, tmp_path, provider_name, host, model, stream):
    settings = setup(wizard, monkeypatch, tmp_path, provider_name, model)
    expected = model or "MiniMax-M3.1-Flash"
    assert settings.provider_model == expected
    await assert_request(monkeypatch, settings, host, expected, stream)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["MiniMax-M2.7", "saved-custom-model", "speech-02-hd"])
async def test_saved_model_is_not_reset(monkeypatch, tmp_path, stream, model):
    saved = tmp_path / ".env"
    saved.write_text(f"PROVIDER_MODEL={model}\nPROVIDER_BASE_URL=https://api.minimax.io/v1\n")
    settings = Settings(_env_file=saved)
    await assert_request(monkeypatch, settings, "api.minimax.io", model, stream)
    assert f"PROVIDER_MODEL={model}\n" in saved.read_text()


@pytest.mark.parametrize("provider_name,model,host,expected", [
    ("Custom", "my-model", "custom.example", "my-model"),
    ("DeepSeek", "", "api.deepseek.com", "deepseek-chat"),
])
async def test_other_setup_paths(wizard, monkeypatch, tmp_path, provider_name, model, host, expected):
    settings = setup(wizard, monkeypatch, tmp_path, provider_name, model)
    await assert_request(monkeypatch, settings, host, expected, False)


async def assert_request(monkeypatch, settings, host, model, stream):
    captured = []

    def respond(request):
        captured.append(request)
        if stream:
            chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
            return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        })

    real_client = httpx.AsyncClient

    def mock_client(**kwargs):
        return real_client(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr("proxy.client.httpx.AsyncClient", mock_client)
    provider = ProviderClient(settings)
    monkeypatch.setattr(routes, "_client", provider)
    monkeypatch.setattr(routes, "count_messages", lambda *args: 10)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        async with real_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/messages", json={
                "model": "client-model", "stream": stream, "max_tokens": 1024,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}},
                ]}],
                "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "auto"},
            })
        assert response.status_code == 200
        assert "ok" in response.text
        assert len(captured) == 1
        request = captured[0]
        assert str(request.url) == f"https://{host}/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == model
        assert payload["stream"] is stream
        assert payload["max_tokens"] == 1024
        assert payload["tools"][0]["function"]["name"] == "lookup"
        assert payload["tool_choice"] == "auto"
        assert payload["messages"][0]["content"][1] == {
            "type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        }
        # The proxy exposes no thinking/effort controls: preserve provider defaults.
        assert not {"thinking", "reasoning_effort", "reasoning_split", "service_tier"} & payload.keys()
        if stream:
            assert payload["stream_options"] == {"include_usage": True}
    finally:
        await provider.aclose()


def test_flash_output_price_does_not_match_m3_prefix():
    cost = load_module("cost")
    assert cost.price_for("MiniMax-M3.1-Flash") == 0.60
    assert cost.price_for("MiniMax-M3") == 2.40
    assert cost.price_for("MiniMax-M2.7") == 1.20
    assert cost.price_for("custom/MiniMax-M3.1-Flash") == cost.FALLBACK_PRICE
    assert cost.price_for("MiniMax-M3-custom") == cost.FALLBACK_PRICE
