"""Route a task to the cheapest OpenAI-compatible model that can answer it.

The router is a two-step process: a cheap classifier decides between a "fast"
and a "powerful" route, then an Agno agent built on a native OpenAI-compatible
endpoint answers the task with the model of that route.

Two classifier backends are available:

* ``openai`` (default) asks any OpenAI-compatible ``/chat/completions`` endpoint
  for a single token and compares the logprobs of the labels "A" and "B".
* ``laya`` calls a local Laya decision model. This is a local model runtime and
  therefore the one deliberate exception to the OpenAI-compatible-only rule.
  Laya is never imported unless this backend is selected.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict, cast

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
from httpx import Client
from openai import BadRequestError, NotFoundError, OpenAI, UnprocessableEntityError
from openai.types.chat import ChatCompletion

os.environ.setdefault("AGNO_TELEMETRY", "false")

load_dotenv()

Route = Literal["fast", "powerful"]
ClassifierName = Literal["openai", "laya"]

TaskInput = str | Sequence[str | Mapping[str, Any]]

# A -> fast model
# B -> powerful model
_LABELS: tuple[str, ...] = ("A", "B")
_ROUTES: tuple[Route, ...] = ("fast", "powerful")
_ROUTE_BY_LABEL: dict[str, Route] = dict(zip(_LABELS, _ROUTES, strict=True))
_REQUIRED_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ROUTER_MODEL",
    "FAST_MODEL",
    "FAST_BASE_URL",
    "FAST_API_KEY",
    "POWERFUL_MODEL",
    "POWERFUL_BASE_URL",
    "POWERFUL_API_KEY",
)

# The router fails closed: an undecidable task goes to the cheaper model.
_FALLBACK_ROUTE: Route = "fast"

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})

# Description shown to the router for each available route.
_CRITERIA: dict[Route, str] = {
    "fast": "Direct lookups, extraction, and localized changes.",
    "powerful": "Architecture and high-stakes decisions.",
}

_INSTRUCTIONS = "Choose the least costly model that can complete the task."

_QUESTION_KEY = "route"

_CHOICE_QUESTION: dict[str, dict[str, object]] = {
    _QUESTION_KEY: {
        "type": "choice",
        "instructions": _INSTRUCTIONS,
        "criteria": {
            label: _CRITERIA[route] for label, route in _ROUTE_BY_LABEL.items()
        },
    }
}

_LABEL_PATTERN = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")


class ConfigError(RuntimeError):
    """Raised when the configuration is incomplete or unusable."""


class Classification(TypedDict):
    route: Route
    confidence: float | None
    source: ClassifierName


class RouteResult(Classification):
    model: str
    content: str


class Classifier(Protocol):
    def predict(self, task: str) -> Classification: ...


class LayaPredict(Protocol):
    """The ``laya.Router.predict`` signature this router depends on."""

    def __call__(
        self, state: str, questions: Mapping[str, Any], **kwargs: Any
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class Endpoint:
    """One model on its own OpenAI-compatible base URL with its own key."""

    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    """Everything the router needs, resolved from the environment.

    The classifier has its own endpoint, and every route has its own endpoint on
    top of that, so a route can sit behind a different gateway or credential.
    """

    router_base_url: str
    router_api_key: str
    router_model: str
    fast: Endpoint
    powerful: Endpoint
    classifier: ClassifierName = "openai"
    laya_model: str = "multilingual"
    laya_max_len: int = 1024
    laya_preload: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        values = {name: os.environ.get(name, "").strip() for name in _REQUIRED_VARS}
        for name, value in values.items():
            if not value:
                raise ConfigError(f"{name} is not set")

        name = os.environ.get("CLASSIFIER", "openai").strip() or "openai"
        if name not in ("openai", "laya"):
            raise ConfigError(f"CLASSIFIER must be 'openai' or 'laya', got {name!r}")

        return cls(
            router_base_url=values["OPENAI_BASE_URL"],
            router_api_key=values["OPENAI_API_KEY"],
            router_model=values["ROUTER_MODEL"],
            fast=Endpoint(
                values["FAST_BASE_URL"], values["FAST_API_KEY"], values["FAST_MODEL"]
            ),
            powerful=Endpoint(
                values["POWERFUL_BASE_URL"],
                values["POWERFUL_API_KEY"],
                values["POWERFUL_MODEL"],
            ),
            classifier=cast(ClassifierName, name),
            laya_model=os.environ.get("LAYA_MODEL", "").strip() or "multilingual",
            laya_max_len=_positive_int(os.environ, "LAYA_MAX_LEN", 1024),
            laya_preload=os.environ.get("LAYA_PRELOAD", "").strip().lower() in _TRUTHY,
        )

    def endpoint_for(self, route: Route) -> Endpoint:
        return self.fast if route == "fast" else self.powerful


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, "").strip()

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from error

    if value <= 0:
        raise ConfigError(f"{name} must be greater than 0, got {value}")

    return value


def normalize_task(task: TaskInput) -> str:
    """Flatten a prompt into plain text, keeping only the text parts."""

    parts: list[str]
    if isinstance(task, str):
        parts = [task]
    else:
        flattened: list[str] = []
        for block in task:
            if isinstance(block, str):
                flattened.append(block)
            elif block.get("type") == "text":
                flattened.append(str(block.get("text", "")))
        parts = flattened

    text = "\n".join(part for part in parts if part.strip())

    if not text.strip():
        raise ValueError("task must contain non-empty text")

    return text


def _options() -> str:
    return "\n".join(
        f"{label}. {_CRITERIA[route]}" for label, route in _ROUTE_BY_LABEL.items()
    )


def _prompt(task: str) -> str:
    return f"""{_INSTRUCTIONS}
Choose one option.

Task: {task}

{_options()}"""


def _from_label(
    label: str, confidence: float | None, source: ClassifierName
) -> Classification:
    return {
        "route": _ROUTE_BY_LABEL.get(label, _FALLBACK_ROUTE),
        "confidence": confidence,
        "source": source,
    }


def _from_text(content: str | None) -> Classification:
    match = _LABEL_PATTERN.search(content or "")
    return _from_label(match.group(1) if match else "", None, "openai")


class OpenAICompatibleClassifier:
    """Ask an OpenAI-compatible endpoint which of the two labels fits best."""

    def __init__(
        self, settings: Settings, *, http_client: Client | None = None
    ) -> None:
        self._client = OpenAI(
            api_key=settings.router_api_key,
            base_url=settings.router_base_url,
            # The openai SDK annotates this as httpx2.Client but only calls the
            # shared httpx surface on it; agno requires a real httpx.Client.
            http_client=cast("Any", http_client),
            max_retries=0,
        )
        self._model = settings.router_model

    def predict(self, task: str) -> Classification:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _prompt(task)}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
            )
        except (BadRequestError, NotFoundError, UnprocessableEntityError):
            return self._predict_without_logprobs(task)

        return self._from_response(response)

    def _predict_without_logprobs(self, task: str) -> Classification:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": _prompt(task)}],
            max_tokens=1,
        )
        return _from_text(response.choices[0].message.content)

    def _from_response(self, response: ChatCompletion) -> Classification:
        choice = response.choices[0]
        logprobs = choice.logprobs
        if logprobs is None or not logprobs.content:
            return _from_text(choice.message.content)

        scores: dict[str, float] = {}
        for entry in logprobs.content[0].top_logprobs or ():
            label = entry.token.strip()
            if label in _ROUTE_BY_LABEL:
                scores[label] = max(scores.get(label, entry.logprob), entry.logprob)

        if not scores:
            return _from_label("", None, "openai")

        best = max(scores.values())
        total = sum(math.exp(score - best) for score in scores.values())
        label = max(scores.items(), key=lambda item: item[1])[0]

        return _from_label(label, 1.0 / total, "openai")


class LayaClassifier:
    """Ask a local Laya decision model which of the two labels fits best."""

    def __init__(
        self,
        *,
        predict: LayaPredict,
        model: str = "multilingual",
        max_len: int = 1024,
    ) -> None:
        self._predict = predict
        self._model = model
        self._max_len = max_len

    @classmethod
    def from_laya(
        cls,
        *,
        model: str = "multilingual",
        preload: bool = False,
        max_len: int = 1024,
    ) -> LayaClassifier:
        try:
            from laya import Router
        except ImportError as error:
            raise ConfigError(
                "the laya classifier needs the optional 'laya' package: "
                "uv sync --extra laya"
            ) from error

        return cls(
            predict=Router(preload=preload).predict,
            model=model,
            max_len=max_len,
        )

    def predict(self, task: str) -> Classification:
        answer = self._answer(task)
        confidence = answer.get("confidence")

        return {
            "route": _ROUTE_BY_LABEL.get(
                str(answer.get("choice", "")).strip(), _FALLBACK_ROUTE
            ),
            "confidence": float(confidence)
            if isinstance(confidence, int | float)
            else None,
            "source": "laya",
        }

    def _answer(self, task: str) -> Mapping[str, Any]:
        payload = self._predict(
            task, _CHOICE_QUESTION, model=self._model, max_len=self._max_len
        )
        answers = payload.get("answers") or {}

        return answers.get(_QUESTION_KEY) or {}


def build_classifier(
    settings: Settings,
    *,
    http_client: Client | None = None,
    laya_predict: LayaPredict | None = None,
) -> Classifier:
    if settings.classifier == "laya":
        if laya_predict is not None:
            return LayaClassifier(
                predict=laya_predict,
                model=settings.laya_model,
                max_len=settings.laya_max_len,
            )
        return LayaClassifier.from_laya(
            model=settings.laya_model,
            max_len=settings.laya_max_len,
            preload=settings.laya_preload,
        )

    return OpenAICompatibleClassifier(settings, http_client=http_client)


class ModelRouter:
    """Classify a task, then answer it with the model of the chosen route."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: Client | None = None,
        classifier: Classifier | None = None,
        laya_predict: LayaPredict | None = None,
    ) -> None:
        self._settings = settings
        self._classifier = classifier or build_classifier(
            settings,
            http_client=http_client,
            laya_predict=laya_predict,
        )
        self._agents = {
            route: Agent(
                name=f"{route}-agent",
                model=OpenAIChat(
                    id=endpoint.model,
                    api_key=endpoint.api_key,
                    base_url=endpoint.base_url,
                    http_client=http_client,
                    max_retries=0,
                ),
            )
            for route, endpoint in (
                (route, settings.endpoint_for(route)) for route in _ROUTES
            )
        }

    def classification(self, task: str) -> Classification:
        return self._classifier.predict(task)

    def classify(self, task: str) -> Route:
        return self._classifier.predict(task)["route"]

    def run(self, task: TaskInput) -> RouteResult:
        text = normalize_task(task)
        decision = self._classifier.predict(text)
        route = decision["route"]

        return {
            **decision,
            "model": self._settings.endpoint_for(route).model,
            "content": str(self._agents[route].run(text).content or ""),
        }


def route_and_run(task: TaskInput) -> RouteResult:
    return ModelRouter(Settings.from_env()).run(task)


if __name__ == "__main__":
    router = ModelRouter(Settings.from_env())

    for prompt in (
        "Extract the city name from: I landed in Lisbon last night.",
        "Propose a short service architecture for a multi-tenant billing system.",
    ):
        outcome = router.run(prompt)

        print(f"task:       {prompt}")
        print(
            f"route:      {outcome['route']} ({outcome['source']}, {outcome['confidence']})"
        )
        print(f"model:      {outcome['model']}")
        print(f"reply:      {outcome['content']}")
        print()
