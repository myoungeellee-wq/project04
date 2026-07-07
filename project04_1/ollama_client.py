from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OllamaConnectionStatus:
    ok: bool
    message: str
    models: list[str]


def normalize_ollama_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        return "http://127.0.0.1:11434"
    if not value.startswith(("http://", "https://")):
        return f"http://{value}"
    return value


def build_ollama_headers(api_key: str = "") -> dict[str, str]:
    token = api_key.strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def fetch_ollama_models(base_url: str, api_key: str = "", timeout: float = 5.0) -> list[str]:
    normalized_url = normalize_ollama_base_url(base_url)
    request = Request(
        urljoin(f"{normalized_url}/", "api/tags"),
        headers=build_ollama_headers(api_key),
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]


def fetch_openai_compatible_models(base_url: str, api_key: str = "", timeout: float = 5.0) -> list[str]:
    normalized_url = normalize_ollama_base_url(base_url)
    request = Request(
        urljoin(f"{normalized_url}/", "v1/models"),
        headers=build_ollama_headers(api_key),
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return [model.get("id", "") for model in payload.get("data", []) if model.get("id")]


def check_ollama_connection(
    base_url: str,
    model_name: str,
    api_key: str = "",
    timeout: float = 5.0,
) -> OllamaConnectionStatus:
    normalized_url = normalize_ollama_base_url(base_url)
    try:
        try:
            models = fetch_ollama_models(normalized_url, api_key=api_key, timeout=timeout)
        except HTTPError:
            models = fetch_openai_compatible_models(normalized_url, api_key=api_key, timeout=timeout)
    except HTTPError as exc:
        return OllamaConnectionStatus(False, f"Ollama HTTP 오류: {exc.code} {exc.reason}", [])
    except URLError as exc:
        return OllamaConnectionStatus(False, f"Ollama 연결 실패: {exc.reason}", [])
    except TimeoutError:
        return OllamaConnectionStatus(False, "Ollama 연결 시간이 초과되었습니다.", [])
    except Exception as exc:
        return OllamaConnectionStatus(False, f"Ollama 확인 실패: {exc}", [])

    if not models:
        return OllamaConnectionStatus(True, f"{normalized_url} 연결 성공. 등록된 모델은 없습니다.", [])

    if model_name in models:
        return OllamaConnectionStatus(True, f"{normalized_url} 연결 성공. `{model_name}` 모델을 사용할 수 있습니다.", models)

    return OllamaConnectionStatus(
        False,
        f"{normalized_url} 연결 성공. 하지만 `{model_name}` 모델이 없습니다.",
        models,
    )
