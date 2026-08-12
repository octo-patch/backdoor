import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("cost", Path(__file__).parents[1] / "cost.py")
assert SPEC and SPEC.loader
cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cost)


def test_parses_complete_and_stream_cache_usage(tmp_path: Path):
    log_path = tmp_path / "proxy.log"
    log_path.write_text(
        "2026-08-12 10:00:00,000 INFO proxy.routes: "
        "← MiniMax-M2.7 [complete] stop=end_turn out_tokens=300 in_tokens=1200 "
        "cache_read_tokens=800\n"
        "2026-08-12 10:01:00,000 INFO proxy.routes: "
        "← MiniMax-M3 [stream] done in_tokens=1000 cache_read_tokens=200\n"
    )

    models = cost.parse_log(log_path)["models"]

    assert models["MiniMax-M2.7"] == {
        "complete": 1,
        "stream": 0,
        "out_tokens": 300,
        "in_tokens": 1200,
        "cache_read_tokens": 800,
    }
    assert models["MiniMax-M3"]["stream"] == 1
    assert models["MiniMax-M3"]["cache_read_tokens"] == 200


@pytest.mark.parametrize(
    ("model", "usage", "expected"),
    [
        (
            "MiniMax-M3",
            {"in_tokens": 1_000_000, "out_tokens": 1_000_000, "cache_read_tokens": 250_000},
            2.88,
        ),
        (
            "MiniMax-M2.7",
            {"in_tokens": 1_000_000, "out_tokens": 1_000_000, "cache_read_tokens": 500_000},
            1.38,
        ),
    ],
)
def test_estimates_input_output_and_cache_read_cost(model, usage, expected):
    assert cost.estimate_cost(model, usage) == pytest.approx(expected)
