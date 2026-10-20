"""Use the real request, logging, and report paths with an offline HTTP transport."""

import importlib.util
import json
import logging
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from proxy import routes
from proxy.client import ProviderClient
from proxy.config import Settings, get_settings


SPEC = importlib.util.spec_from_file_location("cost", Path(__file__).parents[1] / "cost.py")
cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cost)


@pytest.mark.parametrize("host", ["api.minimax.io", "api.minimaxi.com"])
@pytest.mark.parametrize("stream", [False, True])
async def test_flash_http_usage_reaches_cost_report(monkeypatch, tmp_path, host, stream):
    usage = {"prompt_tokens": 1200, "completion_tokens": 300,
             "prompt_tokens_details": {"cached_tokens": 800}}
    requests = []

    def respond(request):
        requests.append(request)
        if not stream:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": usage,
            })
        chunks = [
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            # Usage often arrives after the finish chunk, with no choices.
            {"choices": [], "usage": usage},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    real_client = httpx.AsyncClient
    monkeypatch.setattr("proxy.client.httpx.AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(respond)))
    settings = Settings(_env_file=None, provider_base_url=f"https://{host}/v1",
                        provider_model="MiniMax-M3.1-Flash", provider_api_key="offline-test-key")
    provider = ProviderClient(settings)
    monkeypatch.setattr(routes, "_client", provider)
    monkeypatch.setattr(routes, "count_messages", lambda *args: 999)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_settings] = lambda: settings
    log = tmp_path / "proxy.log"
    handler = logging.FileHandler(log)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    monkeypatch.setattr(routes.logger, "level", logging.INFO)
    routes.logger.addHandler(handler)
    try:
        async with real_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/messages", json={
                "model": "client-model", "stream": stream,
                "messages": [{"role": "user", "content": "Explain the result"}],
            })
        assert response.status_code == 200
        assert "ok" in response.text
        assert len(requests) == 1
        assert str(requests[0].url) == f"https://{host}/v1/chat/completions"
        payload = json.loads(requests[0].content)
        assert payload["model"] == "MiniMax-M3.1-Flash"
        assert payload["stream"] is stream
        if stream:
            assert payload["stream_options"] == {"include_usage": True}
        handler.flush()
        recorded = cost.parse_log(log)["models"]["MiniMax-M3.1-Flash"]
        assert recorded["in_tokens"] == 1200
        assert recorded["out_tokens"] == 300
        assert recorded["cache_read_tokens"] == 800
        assert cost.estimate_cost("MiniMax-M3.1-Flash", recorded) == pytest.approx(0.000252)
    finally:
        routes.logger.removeHandler(handler)
        handler.close()
        await provider.aclose()


@pytest.mark.parametrize("model", ["custom/MiniMax-M3.1-Flash", "MiniMax-M3-custom"])
def test_custom_models_do_not_inherit_direct_prices(model):
    usage = {"in_tokens": 1_000_000, "out_tokens": 1_000_000, "cache_read_tokens": 0}
    assert cost.price_for(model) == cost.FALLBACK_PRICE
    assert cost.estimate_cost(model, usage) == cost.FALLBACK_PRICE


def test_flash_price_is_not_m3_price():
    assert cost.price_for("MiniMax-M3.1-Flash") == 0.60
    assert cost.price_for("MiniMax-M3") == 2.40
    assert cost.price_for("MiniMax-M2.7") == 1.20


def test_m3_report_labels_conservative_standard_estimate(tmp_path, monkeypatch, capsys):
    log = tmp_path / "proxy.log"
    log.write_text("2026-10-20 10:00:00,000 INFO proxy.routes: "
                   "← MiniMax-M3 [complete] stop=end_turn out_tokens=1 in_tokens=10\n")
    monkeypatch.setattr("sys.argv", ["cost.py", "--log", str(log)])
    cost.main()
    output = capsys.readouterr().out
    assert ">512k standard rate as an upper estimate" in output
    assert "per-request input tiers are not retained" in output
