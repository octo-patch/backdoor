from proxy.translate import stream_openai_to_anthropic


def test_stream_usage_chunk_records_input_and_cache_tokens():
    state = {"started": True, "block_index": 0, "tool_calls": {}}
    chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 800},
        },
    }

    events = stream_openai_to_anthropic(chunk, state, "msg_test", None, 1000)

    assert events == []
    assert state["input_tokens"] == 1200
    assert state["cache_read_tokens"] == 800
