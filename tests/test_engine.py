from __future__ import annotations

import pytest

from ivonar_inference.engine import Engine, GenerationSettings, _cut_at_stop_strings

from conftest import ANSWER, fake_stream


def test_stream_renders_the_conversation_and_reports_usage(engine: Engine) -> None:
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Where is Paris?"},
    ]
    text = "".join(engine.stream(messages))
    assert text == ANSWER
    call = fake_stream.calls[-1]
    assert call["prompt"] == (
        "<|im_start|><|system|>Be brief.<|im_end|>"
        "<|im_start|><|user|>Hi<|im_end|>"
        "<|im_start|><|assistant|>Hello.<|im_end|>"
        "<|im_start|><|user|>Where is Paris?<|im_end|>"
        "<|im_start|><|assistant|>"
    )
    assert call["stop_token_ids"] == (3,)
    assert call["max_new_tokens"] == 32
    result = engine.last_generation
    assert result.completion_tokens == len(ANSWER.split())
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == engine.count_tokens(call["prompt"])


def test_an_abandoned_stream_releases_the_engine(engine: Engine) -> None:
    stream = engine.stream([{"role": "user", "content": "Where is Paris?"}])
    assert next(stream) == "Paris"
    stream.close()
    result = engine.complete([{"role": "user", "content": "Where is Paris?"}])
    assert result.text == ANSWER
    assert result.finish_reason == "stop"


def test_complete_returns_the_result_with_text(engine: Engine) -> None:
    result = engine.complete([{"role": "user", "content": "Where is Paris?"}])
    assert result.text == ANSWER
    assert result.tokens_per_second >= 0.0


def test_max_tokens_marks_a_cut_answer_as_length(engine: Engine) -> None:
    result = engine.complete([{"role": "user", "content": "Where is Paris?"}], GenerationSettings(max_tokens=3))
    assert result.text == "Paris is the"
    assert result.finish_reason == "length"


def test_settings_are_validated_and_capped_by_the_context() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GenerationSettings(temperature=0).validated(64)
    with pytest.raises(ValueError, match="max_tokens"):
        GenerationSettings(max_tokens=0).validated(64)
    with pytest.raises(ValueError, match="stop"):
        GenerationSettings(stop=("",)).validated(64)
    assert GenerationSettings(max_tokens=500).validated(64).max_tokens == 63


def test_prompt_drops_the_oldest_turns_to_fit_the_context(engine: Engine) -> None:
    long_turn = "word " * 20
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": long_turn},
        {"role": "assistant", "content": long_turn},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "last"},
    ]
    prompt, prompt_tokens = engine.build_prompt(messages, GenerationSettings(max_tokens=32))
    assert prompt_tokens + 32 <= engine.info.context_tokens
    assert "Be brief." in prompt
    assert "second" in prompt
    assert "word word" not in prompt
    assert prompt.endswith("<|im_start|><|user|>last<|im_end|><|im_start|><|assistant|>")


def test_prompt_that_cannot_fit_is_refused(engine: Engine) -> None:
    messages = [{"role": "user", "content": "word " * 80}]
    with pytest.raises(ValueError, match="leaves no room"):
        engine.build_prompt(messages, GenerationSettings(max_tokens=32))


def test_stop_strings_cut_the_stream_before_the_match() -> None:
    deltas = iter(["Paris is", " the capital", ". END of", " story"])
    assert "".join(_cut_at_stop_strings(deltas, ("END",))) == "Paris is the capital. "
    deltas = iter(["Par", "is is ", "here"])
    assert "".join(_cut_at_stop_strings(deltas, ("XYZ",))) == "Paris is here"
    deltas = iter(["a", "b", "c"])
    assert "".join(_cut_at_stop_strings(deltas, ())) == "abc"


def test_stop_strings_reach_the_generation(engine: Engine) -> None:
    result = engine.complete(
        [{"role": "user", "content": "Where is Paris?"}],
        GenerationSettings(max_tokens=32, stop=("capital",)),
    )
    assert result.text == "Paris is the "
    assert result.finish_reason == "stop"
