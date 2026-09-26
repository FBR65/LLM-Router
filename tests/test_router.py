"""Behavioral tests for router.py. No network: the endpoint is a mocked transport."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import httpx
import pytest
from openai import APIStatusError

import router as router_module
from router import (
    ConfigError,
    Endpoint,
    LayaClassifier,
    ModelRouter,
    OpenAICompatibleClassifier,
    RouteResult,
    Settings,
    TaskInput,
    build_classifier,
    normalize_task,
    route_and_run,
)

CHAT_PATH = "/v1/chat/completions"

Response = httpx.Response | tuple[int, Mapping[str, Any]] | Mapping[str, Any]
Responder = Response | list[Response] | Callable[[Mapping[str, Any]], Response]


def completion(
    content: str = "A",
    top_logprobs: Sequence[Mapping[str, Any]] | None = None,
    model: str = "router-model",
) -> Mapping[str, Any]:
    """A minimal but schema-correct /chat/completions payload."""
    logprobs: Mapping[str, Any] | None = None
    if top_logprobs is not None:
        logprobs = {
            "content": [
                {
                    "token": content,
                    "logprob": -0.25,
                    "top_logprobs": list(top_logprobs),
                }
            ]
        }
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": logprobs,
            }
        ],
    }


class Call(NamedTuple):
    url: str
    authorization: str
    body: Mapping[str, Any]


class FakeEndpoint:
    """Records every call and replays canned OpenAI-compatible responses."""

    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.calls: list[Call] = []

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == CHAT_PATH, request.url.path
            body = json.loads(request.content)
            self.calls.append(
                Call(
                    url=str(request.url),
                    authorization=request.headers.get("authorization", ""),
                    body=body,
                )
            )
            responder = self.responder
            if callable(responder):
                result: Response = responder(body)
            elif isinstance(responder, list):
                result = responder[len(self.calls) - 1]
            else:
                result = responder
            if isinstance(result, httpx.Response):
                return result
            if isinstance(result, Mapping):
                return httpx.Response(200, json=result)
            status, payload = result
            return httpx.Response(status, json=payload)

        return httpx.Client(transport=httpx.MockTransport(handler))

    @property
    def requests(self) -> list[Mapping[str, Any]]:
        return [call.body for call in self.calls]

    @property
    def models_called(self) -> list[str]:
        return [str(call.body["model"]) for call in self.calls]


def set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ROUTER_MODEL",
        "FAST_MODEL",
        "FAST_BASE_URL",
        "FAST_API_KEY",
        "POWERFUL_MODEL",
        "POWERFUL_BASE_URL",
        "POWERFUL_API_KEY",
    ):
        monkeypatch.setenv(name, f"value-for-{name}")


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "router_base_url": "https://router.internal/v1",
        "router_api_key": "router-key",
        "router_model": "router-model",
        "fast": Endpoint("https://fast.internal/v1", "fast-key", "fast-model"),
        "powerful": Endpoint(
            "https://powerful.internal/v1", "powerful-key", "powerful-model"
        ),
        "classifier": "openai",
        "laya_model": "multilingual",
        "laya_max_len": 1024,
    }
    base.update(overrides)
    return Settings(**base)


# --- S2: configuration ------------------------------------------------------


def test_settings_reads_every_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "OPENAI_API_KEY": "router-key",
        "OPENAI_BASE_URL": "https://router.internal/v1",
        "ROUTER_MODEL": "router-model",
        "FAST_MODEL": "fast-model",
        "FAST_BASE_URL": "https://fast.internal/v1",
        "FAST_API_KEY": "fast-key",
        "POWERFUL_MODEL": "powerful-model",
        "POWERFUL_BASE_URL": "https://powerful.internal/v1",
        "POWERFUL_API_KEY": "powerful-key",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert Settings.from_env() == Settings(
        router_base_url="https://router.internal/v1",
        router_api_key="router-key",
        router_model="router-model",
        fast=Endpoint("https://fast.internal/v1", "fast-key", "fast-model"),
        powerful=Endpoint(
            "https://powerful.internal/v1", "powerful-key", "powerful-model"
        ),
    )


@pytest.mark.parametrize(
    "missing",
    [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ROUTER_MODEL",
        "FAST_MODEL",
        "FAST_BASE_URL",
        "FAST_API_KEY",
        "POWERFUL_MODEL",
        "POWERFUL_BASE_URL",
        "POWERFUL_API_KEY",
    ],
)
def test_settings_rejects_missing_env_var(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    set_required_env(monkeypatch)
    monkeypatch.delenv(missing)

    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env()

    assert missing in str(excinfo.value)


@pytest.mark.parametrize(
    "blank", ["FAST_MODEL", "FAST_BASE_URL", "FAST_API_KEY", "POWERFUL_API_KEY"]
)
def test_settings_rejects_blank_env_var(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    set_required_env(monkeypatch)
    monkeypatch.setenv(blank, "   ")

    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env()

    assert blank in str(excinfo.value)


def test_settings_rejects_unknown_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required_env(monkeypatch)
    monkeypatch.setenv("CLASSIFIER", "ollama")

    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env()

    assert "CLASSIFIER" in str(excinfo.value)


def test_settings_exposes_the_endpoint_of_a_route() -> None:
    config = settings()

    assert config.endpoint_for("fast") is config.fast
    assert config.endpoint_for("powerful") is config.powerful


# --- S2b: classifier selection ---------------------------------------------


def test_settings_reads_classifier_and_laya_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_env(monkeypatch)
    monkeypatch.setenv("CLASSIFIER", "laya")
    monkeypatch.setenv("LAYA_MODEL", "multilingual")

    assert Settings.from_env() == settings(
        router_base_url="value-for-OPENAI_BASE_URL",
        router_api_key="value-for-OPENAI_API_KEY",
        router_model="value-for-ROUTER_MODEL",
        fast=Endpoint(
            "value-for-FAST_BASE_URL", "value-for-FAST_API_KEY", "value-for-FAST_MODEL"
        ),
        powerful=Endpoint(
            "value-for-POWERFUL_BASE_URL",
            "value-for-POWERFUL_API_KEY",
            "value-for-POWERFUL_MODEL",
        ),
        classifier="laya",
        laya_model="multilingual",
    )


def test_settings_defaults_to_openai_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_env(monkeypatch)
    monkeypatch.delenv("CLASSIFIER", raising=False)
    monkeypatch.delenv("LAYA_MODEL", raising=False)

    assert Settings.from_env().classifier == "openai"
    assert Settings.from_env().laya_model == "multilingual"
    assert Settings.from_env().laya_max_len == 1024
    assert Settings.from_env().laya_preload is False


# --- S3: classification -----------------------------------------------------


def test_classify_picks_fast_when_a_wins() -> None:
    endpoint = FakeEndpoint(
        completion(
            top_logprobs=[
                {"token": "A", "logprob": -0.1},
                {"token": "B", "logprob": -4.0},
            ]
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    assert router.classify("Extract the city name.") == "fast"


def test_classify_picks_powerful_when_b_wins() -> None:
    endpoint = FakeEndpoint(
        completion(
            top_logprobs=[
                {"token": "A", "logprob": -5.0},
                {"token": "B", "logprob": -0.2},
            ]
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    assert router.classify("Design the billing architecture.") == "powerful"


def test_classify_request_targets_router_model_with_one_token() -> None:
    endpoint = FakeEndpoint(completion(top_logprobs=[{"token": "A", "logprob": -0.1}]))
    router = ModelRouter(settings(), http_client=endpoint.client())

    router.classify("Extract the city name.")

    request = endpoint.requests[0]
    assert request["model"] == "router-model"
    assert request["max_tokens"] == 1
    assert request["logprobs"] is True
    prompt = "\n".join(str(part) for part in request["messages"])
    assert "Task: Extract the city name." in prompt
    assert "A. Direct lookups, extraction, and localized changes." in prompt
    assert "B. Architecture and high-stakes decisions." in prompt
    assert "Choose the least costly model that can complete the task." in prompt


def test_classify_matches_space_prefixed_label_token() -> None:
    endpoint = FakeEndpoint(
        completion(
            top_logprobs=[
                {"token": " A", "logprob": -0.1},
                {"token": " B", "logprob": -3.0},
            ]
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    assert router.classify("task") == "fast"


def test_classify_ignores_unrelated_tokens() -> None:
    endpoint = FakeEndpoint(
        completion(
            top_logprobs=[
                {"token": "A", "logprob": -2.0},
                {"token": "The", "logprob": -0.1},
                {"token": "B", "logprob": -0.5},
                {"token": "Maybe", "logprob": -0.2},
            ]
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    assert router.classify("task") == "powerful"


def test_classify_falls_back_to_content_when_logprobs_rejected() -> None:
    calls: list[Mapping[str, Any]] = []

    def responder(body: Mapping[str, Any]) -> Response:
        calls.append(body)
        if body.get("logprobs"):
            return (400, {"error": {"message": "logprobs unsupported"}})
        return completion(content="B")

    endpoint = FakeEndpoint(responder)
    router = ModelRouter(settings(), http_client=endpoint.client())

    assert router.classify("Design the billing architecture.") == "powerful"
    assert calls[0]["logprobs"] is True
    assert calls[1].get("logprobs") is None


def test_classify_defaults_to_fast_on_unparsable_content() -> None:
    def responder(body: Mapping[str, Any]) -> Response:
        if body.get("logprobs"):
            return (400, {"error": {"message": "logprobs unsupported"}})
        return completion(content="I cannot help with that")

    router = ModelRouter(settings(), http_client=FakeEndpoint(responder).client())

    assert router.classify("task") == "fast"


def test_classify_fails_closed_to_fast_when_top_logprobs_are_empty() -> None:
    # Logprobs are present but contain neither label: fall back to the
    # cheapest route instead of trusting an unrelated token.
    endpoint = FakeEndpoint(completion(content="B", top_logprobs=[]))
    router = ModelRouter(settings(), http_client=endpoint.client())

    assert router.classify("task") == "fast"


def test_classify_uses_content_when_logprobs_are_absent() -> None:
    endpoint = FakeEndpoint(completion(content="B", top_logprobs=None))
    router = ModelRouter(settings(), http_client=endpoint.client())

    assert router.classify("task") == "powerful"


def test_classifier_calls_its_own_endpoint_with_its_own_key() -> None:
    endpoint = FakeEndpoint(completion(top_logprobs=[{"token": "A", "logprob": -0.1}]))
    router = ModelRouter(settings(), http_client=endpoint.client())

    router.classify("task")

    call = endpoint.calls[0]
    assert call.url == "https://router.internal/v1/chat/completions"
    assert call.authorization == "Bearer router-key"


def test_fast_route_uses_its_own_base_url_and_api_key() -> None:
    endpoint = FakeEndpoint(
        lambda body: completion(
            content="A" if body["model"] == "router-model" else "Lisbon",
            top_logprobs=(
                [{"token": "A", "logprob": -0.1}]
                if body["model"] == "router-model"
                else None
            ),
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    result = router.run("Extract the city name from: I landed in Lisbon.")

    assert result["model"] == "fast-model"
    answer = endpoint.calls[1]
    assert answer.url == "https://fast.internal/v1/chat/completions"
    assert answer.authorization == "Bearer fast-key"


def test_powerful_route_uses_its_own_base_url_and_api_key() -> None:
    endpoint = FakeEndpoint(
        lambda body: completion(
            content="B" if body["model"] == "router-model" else "Modular monolith.",
            top_logprobs=(
                [{"token": "B", "logprob": -0.1}]
                if body["model"] == "router-model"
                else None
            ),
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    result = router.run("Design the billing architecture.")

    assert result["model"] == "powerful-model"
    answer = endpoint.calls[1]
    assert answer.url == "https://powerful.internal/v1/chat/completions"
    assert answer.authorization == "Bearer powerful-key"


def test_the_two_routes_never_share_an_endpoint() -> None:
    answers = FakeEndpoint(
        lambda body: completion(
            content="A" if "Extract" in str(body["messages"]) else "B",
            top_logprobs=(
                [
                    {
                        "token": "A" if "Extract" in str(body["messages"]) else "B",
                        "logprob": -0.1,
                    }
                ]
                if body["model"] == "router-model"
                else None
            ),
        )
    )
    router = ModelRouter(settings(), http_client=answers.client())

    router.run("Extract the city name.")
    router.run("Design the billing architecture.")

    routes = {
        call.url: call.authorization
        for call in answers.calls
        if call.body["model"] != "router-model"
    }
    assert routes == {
        "https://fast.internal/v1/chat/completions": "Bearer fast-key",
        "https://powerful.internal/v1/chat/completions": "Bearer powerful-key",
    }


# --- S5: task normalization -------------------------------------------------


def test_normalize_task_returns_plain_string() -> None:
    assert normalize_task("Extract the city name.") == "Extract the city name."


def test_normalize_task_joins_text_blocks() -> None:
    task: TaskInput = [
        "first",
        {"type": "text", "text": "second"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
        "third",
    ]

    assert normalize_task(task) == "first\nsecond\nthird"


def test_normalize_task_rejects_text_block_without_text() -> None:
    with pytest.raises(ValueError):
        normalize_task([{"type": "text"}])


@pytest.mark.parametrize("task", ["", "   ", [], ["", "  "], [{"type": "image_url"}]])
def test_normalize_task_rejects_empty_input(task: Any) -> None:
    with pytest.raises(ValueError):
        normalize_task(task)


# --- S4: routing + agent execution ------------------------------------------


def test_run_answers_with_the_fast_model() -> None:
    endpoint = FakeEndpoint(
        lambda body: completion(
            content="A" if body["model"] == "router-model" else "Lisbon",
            top_logprobs=(
                [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -4.0}]
                if body["model"] == "router-model"
                else None
            ),
            model=str(body["model"]),
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    result: RouteResult = router.run("Extract the city name from: I landed in Lisbon.")

    assert result["route"] == "fast"
    assert result["model"] == "fast-model"
    assert result["content"] == "Lisbon"
    assert result["source"] == "openai"
    assert result["confidence"] is not None
    assert result["confidence"] > 0.95
    assert endpoint.models_called == ["router-model", "fast-model"]


def test_run_answers_with_the_powerful_model() -> None:
    endpoint = FakeEndpoint(
        lambda body: completion(
            content="B"
            if body["model"] == "router-model"
            else "Use a modular monolith.",
            top_logprobs=(
                [{"token": "A", "logprob": -5.0}, {"token": "B", "logprob": -0.1}]
                if body["model"] == "router-model"
                else None
            ),
            model=str(body["model"]),
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    result = router.run("Propose a service architecture for multi-tenant billing.")

    assert result["route"] == "powerful"
    assert result["model"] == "powerful-model"
    assert result["content"] == "Use a modular monolith."
    assert result["source"] == "openai"
    assert result["confidence"] is not None
    assert result["confidence"] > 0.95
    assert endpoint.models_called == ["router-model", "powerful-model"]


def test_run_rejects_empty_task_before_any_request() -> None:
    endpoint = FakeEndpoint(completion())
    router = ModelRouter(settings(), http_client=endpoint.client())

    with pytest.raises(ValueError):
        router.run("   ")

    assert endpoint.requests == []


def test_run_passes_the_task_to_the_selected_agent() -> None:
    endpoint = FakeEndpoint(
        lambda body: completion(
            content="A",
            top_logprobs=[{"token": "A", "logprob": -0.1}],
            model=str(body["model"]),
        )
    )
    router = ModelRouter(settings(), http_client=endpoint.client())

    router.run(["Extract the city", {"type": "text", "text": "name."}])

    answer_request = endpoint.requests[1]
    user_turns = [
        str(message["content"])
        for message in answer_request["messages"]
        if message["role"] == "user"
    ]
    assert user_turns == ["Extract the city\nname."]


def test_route_and_run_uses_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://router.internal/v1")
    monkeypatch.setenv("ROUTER_MODEL", "r1")
    monkeypatch.setenv("FAST_MODEL", "f1")
    monkeypatch.setenv("FAST_BASE_URL", "https://fast.internal/v1")
    monkeypatch.setenv("FAST_API_KEY", "fk")
    monkeypatch.setenv("POWERFUL_MODEL", "p1")
    monkeypatch.setenv("POWERFUL_BASE_URL", "https://powerful.internal/v1")
    monkeypatch.setenv("POWERFUL_API_KEY", "pk")
    captured: list[str] = []

    def fake_init(self: ModelRouter, *args: Any, **kwargs: Any) -> None:
        captured.append("init")

    def fake_run(self: ModelRouter, task: Any) -> RouteResult:
        captured.append(task)
        return {
            "route": "fast",
            "model": "f1",
            "content": "ok",
            "confidence": None,
            "source": "openai",
        }

    monkeypatch.setattr(ModelRouter, "__init__", fake_init)
    monkeypatch.setattr(ModelRouter, "run", fake_run)

    assert route_and_run("task") == {
        "route": "fast",
        "model": "f1",
        "content": "ok",
        "confidence": None,
        "source": "openai",
    }
    assert captured == ["init", "task"]


def test_classify_falls_back_to_content_when_route_is_unknown() -> None:
    calls: list[Mapping[str, Any]] = []

    def responder(body: Mapping[str, Any]) -> Response:
        calls.append(body)
        if body.get("logprobs"):
            return (404, {"error": {"message": "no such route"}})
        return completion(content="B")

    router = ModelRouter(settings(), http_client=FakeEndpoint(responder).client())

    assert router.classify("Design the billing architecture.") == "powerful"
    assert len(calls) == 2


def test_classify_propagates_endpoint_errors() -> None:
    endpoint = FakeEndpoint((500, {"error": {"message": "boom"}}))
    router = ModelRouter(settings(), http_client=endpoint.client())

    with pytest.raises(APIStatusError):
        router.classify("task")

    assert len(endpoint.requests) == 1


# --- S2b: classifier selection ---------------------------------------------


def test_build_classifier_defaults_to_openai_compatible() -> None:
    assert isinstance(build_classifier(settings()), OpenAICompatibleClassifier)


def test_build_classifier_selects_laya() -> None:
    assert isinstance(
        build_classifier(settings(classifier="laya"), laya_predict=lambda *a, **k: {}),
        LayaClassifier,
    )


def test_build_classifier_loads_laya_when_no_predict_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def fake_from_laya(**kwargs: Any) -> LayaClassifier:
        recorded.update(kwargs)
        return LayaClassifier(predict=laya_answer("A"))

    monkeypatch.setattr(LayaClassifier, "from_laya", fake_from_laya)

    classifier = build_classifier(
        settings(classifier="laya", laya_model="multilingual")
    )

    assert isinstance(classifier, LayaClassifier)
    assert recorded == {
        "model": "multilingual",
        "preload": False,
        "max_len": 1024,
    }


def test_router_uses_laya_classifier() -> None:
    router = ModelRouter(
        settings(classifier="laya"),
        laya_predict=lambda state, questions, **kwargs: {
            "answers": {"route": {"choice": "B", "confidence": 0.71}}
        },
    )

    assert router.classification("Design the billing architecture.") == {
        "route": "powerful",
        "confidence": 0.71,
        "source": "laya",
    }


def test_run_with_laya_classifier_answers_with_powerful_model() -> None:
    def laya_predict(state: Any, questions: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {"answers": {"route": {"choice": "B", "confidence": 0.9}}}

    endpoint = FakeEndpoint(completion(content="served by powerful-model", model="x"))
    router = ModelRouter(
        settings(classifier="laya"),
        http_client=endpoint.client(),
        laya_predict=laya_predict,
    )

    result = router.run("Design the billing architecture.")

    assert result == {
        "route": "powerful",
        "model": "powerful-model",
        "content": "served by powerful-model",
        "confidence": 0.9,
        "source": "laya",
    }
    assert endpoint.models_called == ["powerful-model"]


# --- S3b: laya backend -----------------------------------------------------


def laya_answer(choice: object) -> Callable[..., Mapping[str, Any]]:
    def predict(state: Any, questions: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {"answers": {"route": {"choice": choice, "confidence": 0.5}}}

    return predict


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("A", "fast"), ("B", "powerful")],
)
def test_laya_classifier_maps_choice_to_route(choice: str, expected: str) -> None:
    classifier = LayaClassifier(predict=laya_answer(choice))

    assert classifier.predict("task")["route"] == expected


def test_laya_classifier_reports_confidence_and_source() -> None:
    classifier = LayaClassifier(
        predict=lambda state, questions, **kwargs: {
            "answers": {"route": {"choice": "A", "confidence": 0.83}}
        }
    )

    assert classifier.predict("task") == {
        "route": "fast",
        "confidence": 0.83,
        "source": "laya",
    }


def test_laya_classifier_sends_typed_choice_question() -> None:
    captured: dict[str, Any] = {}

    def predict(state: str, questions: Any, **kwargs: Any) -> Mapping[str, Any]:
        captured["state"] = state
        captured["questions"] = questions
        captured["kwargs"] = kwargs
        return {"answers": {"route": {"choice": "A"}}}

    classifier = LayaClassifier(predict=predict, model="typed-decisions", max_len=8192)

    classifier.predict("Extract the city name.")

    question = captured["questions"]["route"]
    assert question["type"] == "choice"
    assert question["instructions"] == (
        "Choose the least costly model that can complete the task."
    )
    assert question["criteria"] == {
        "A": "Direct lookups, extraction, and localized changes.",
        "B": "Architecture and high-stakes decisions.",
    }
    assert captured["state"] == "Extract the city name."
    assert captured["kwargs"] == {"model": "typed-decisions", "max_len": 8192}


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"answers": {}},
        {"answers": {"route": {}}},
        {"answers": {"route": {"choice": "C"}}},
        {"answers": {"route": {"choice": None}}},
        {"answers": {"route": {"choice": "A", "confidence": "high"}}},
    ],
)
def test_laya_classifier_fails_closed_to_fast(answer: Mapping[str, Any]) -> None:
    def predict(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return answer

    classifier = LayaClassifier(predict=predict)

    assert classifier.predict("task")["route"] == "fast"


def test_laya_classifier_handles_missing_confidence() -> None:
    classifier = LayaClassifier(
        predict=lambda *args, **kwargs: {"answers": {"route": {"choice": "B"}}}
    )

    assert classifier.predict("task") == {
        "route": "powerful",
        "confidence": None,
        "source": "laya",
    }


def test_laya_classifier_propagates_predict_errors() -> None:
    def predict(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        raise RuntimeError("no checkpoint")

    with pytest.raises(RuntimeError):
        LayaClassifier(predict=predict).predict("task")


def test_laya_classifier_from_laya_uses_laya_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    class FakeLayaRouter:
        def __init__(self, preload: bool = False) -> None:
            recorded["preload"] = preload

        def predict(
            self, state: Any, questions: Any, **kwargs: Any
        ) -> Mapping[str, Any]:
            recorded["called"] = (state, questions, kwargs)
            return {"answers": {"route": {"choice": "B"}}}

    module = types.ModuleType("laya")
    module.Router = FakeLayaRouter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", module)

    classifier = LayaClassifier.from_laya(model="multilingual", preload=True)

    assert recorded["preload"] is True
    assert classifier.predict("task")["route"] == "powerful"
    assert recorded["called"][2] == {"model": "multilingual", "max_len": 1024}


def test_settings_reads_laya_preload(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required_env(monkeypatch)
    monkeypatch.setenv("LAYA_PRELOAD", "true")

    assert Settings.from_env().laya_preload is True

    monkeypatch.setenv("LAYA_PRELOAD", "1")
    assert Settings.from_env().laya_preload is True

    monkeypatch.setenv("LAYA_PRELOAD", "no")
    assert Settings.from_env().laya_preload is False

    monkeypatch.delenv("LAYA_PRELOAD")
    assert Settings.from_env().laya_preload is False


def test_build_classifier_honours_the_preload_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def fake_from_laya(**kwargs: Any) -> LayaClassifier:
        recorded.update(kwargs)
        return LayaClassifier(predict=laya_answer("A"))

    monkeypatch.setattr(LayaClassifier, "from_laya", fake_from_laya)

    build_classifier(settings(classifier="laya", laya_preload=True))

    assert recorded == {"model": "multilingual", "preload": True, "max_len": 1024}


def test_laya_classifier_defaults_to_the_multilingual_checkpoint() -> None:
    captured: dict[str, Any] = {}

    def predict(state: str, questions: Any, **kwargs: Any) -> Mapping[str, Any]:
        captured["kwargs"] = kwargs
        return {"answers": {"route": {"choice": "A"}}}

    LayaClassifier(predict=predict).predict("Extract the city name.")

    assert captured["kwargs"] == {"model": "multilingual", "max_len": 1024}


def test_settings_reads_laya_max_len(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required_env(monkeypatch)
    monkeypatch.setenv("LAYA_MAX_LEN", "8192")

    assert Settings.from_env().laya_max_len == 8192


@pytest.mark.parametrize("value", ["abc", "0", "-1", "10.5"])
def test_settings_rejects_invalid_laya_max_len(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    set_required_env(monkeypatch)
    monkeypatch.setenv("LAYA_MAX_LEN", value)

    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env()

    assert "LAYA_MAX_LEN" in str(excinfo.value)


def test_build_classifier_forwards_laya_max_len() -> None:
    recorded: dict[str, Any] = {}

    def fake_from_laya(**kwargs: Any) -> LayaClassifier:
        recorded.update(kwargs)
        return LayaClassifier(predict=laya_answer("A"))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LayaClassifier, "from_laya", fake_from_laya)
        build_classifier(settings(classifier="laya", laya_max_len=8192))

    assert recorded["max_len"] == 8192


def test_laya_classifier_from_laya_import_error_is_actionable() -> None:
    with pytest.raises(ConfigError) as excinfo:
        LayaClassifier.from_laya()

    assert "laya" in str(excinfo.value)


# --- S6: forbidden dependencies ---------------------------------------------


FORBIDDEN = ("langchain", "langgraph", "llama_cpp", "numpy")


def test_module_does_not_import_forbidden_frameworks() -> None:
    source = (router_module.__file__ or "").replace(".pyc", ".py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    for name in FORBIDDEN:
        assert f"import {name}" not in text, name
        assert f"from {name}" not in text, name


def test_forbidden_frameworks_are_not_loaded() -> None:
    for name in FORBIDDEN:
        assert name not in router_module.__dict__, name


def test_laya_is_not_imported_at_module_import_time() -> None:
    assert "laya" not in router_module.__dict__
    assert router_module.LayaClassifier.__module__ == "router"
