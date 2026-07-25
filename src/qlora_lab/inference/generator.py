from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from qlora_lab.common.errors import AppError
from qlora_lab.inference.schemas import ChatCompletionRequest
from qlora_lab.inference.settings import InferenceSettings

LOGGER = logging.getLogger(__name__)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass(slots=True)
class GenerationResult:
    content: str | None
    tool_calls: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int


class Generator(Protocol):
    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def generate(self, request: ChatCompletionRequest) -> GenerationResult: ...

    def health(self) -> dict[str, Any]: ...


class TransformersGenerator:
    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.device = "unloaded"
        self._generation_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError('Install model dependencies with: pip install -e ".[ml]"') from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to load the configured 4-bit QLoRA model.")
        if not self.settings.adapter_path.is_dir():
            raise RuntimeError(
                f"QLoRA adapter not found at {self.settings.adapter_path}. Run training first."
            )

        adapter_config = PeftConfig.from_pretrained(str(self.settings.adapter_path))
        trained_base = str(adapter_config.base_model_name_or_path or "")
        if (
            trained_base
            and trained_base != self.settings.base_model_id
            and not self.settings.allow_adapter_model_mismatch
        ):
            raise RuntimeError(
                "Adapter/base model mismatch: "
                f"adapter expects {trained_base!r}, configured {self.settings.base_model_id!r}."
            )

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer_source: str | Path = self.settings.adapter_path
        if not (self.settings.adapter_path / "tokenizer_config.json").is_file():
            tokenizer_source = self.settings.base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.settings.base_model_id,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=compute_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        self.model = PeftModel.from_pretrained(base, str(self.settings.adapter_path))
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)
        LOGGER.info(
            "Loaded base model %s with adapter %s on %s",
            self.settings.base_model_id,
            self.settings.adapter_path,
            self.device,
        )

    def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        if not self.loaded:
            raise AppError("MODEL_NOT_READY", "Model is not loaded.", 503, True)

        import torch

        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        tools = (
            [tool.model_dump(exclude_none=True) for tool in request.tools]
            if request.tools
            else None
        )
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_length = int(encoded["input_ids"].shape[-1])
        context_window = _model_context_window(self.model, self.tokenizer)
        requested_tokens = min(request.max_tokens, self.settings.max_new_tokens)
        if context_window is not None and input_length + requested_tokens > context_window:
            raise AppError(
                "CONTEXT_LENGTH_EXCEEDED",
                "Prompt and requested output exceed the model context window.",
                400,
                False,
                {
                    "prompt_tokens": input_length,
                    "requested_output_tokens": requested_tokens,
                    "context_window": context_window,
                },
            )
        encoded = encoded.to(self.device)
        do_sample = request.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": requested_tokens,
            "do_sample": do_sample,
            "top_p": request.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = request.temperature

        try:
            with self._generation_lock, torch.inference_mode():
                output = self.model.generate(**encoded, **generation_kwargs)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise AppError(
                "CUDA_OUT_OF_MEMORY",
                "GPU ran out of memory. Reduce input length or max_tokens.",
                503,
                True,
            ) from exc

        generated_ids = output[0, input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        text = _apply_stop(text, request.stop)
        content, tool_calls = parse_tool_calls(text)
        return GenerationResult(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=input_length,
            completion_tokens=int(generated_ids.shape[-1]),
        )

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "ready" if self.loaded else "starting",
            "model_loaded": self.loaded,
            "base_model": self.settings.base_model_id,
            "adapter": str(self.settings.adapter_path),
            "device": self.device,
        }
        try:
            import torch

            if torch.cuda.is_available():
                result["cuda_memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated(0) / 1024**2, 2
                )
        except ImportError:
            pass
        return result


def _apply_stop(text: str, stop: str | list[str] | None) -> str:
    if not stop:
        return text
    sequences = [stop] if isinstance(stop, str) else stop
    positions = [text.find(item) for item in sequences if item and item in text]
    return text[: min(positions)].rstrip() if positions else text


def _model_context_window(model: Any, tokenizer: Any) -> int | None:
    candidates = (
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    )
    for candidate in candidates:
        if isinstance(candidate, int) and 0 < candidate < 10_000_000:
            return candidate
    return None


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("Model emitted invalid tool JSON: %s", raw[:500])
            continue
        if not isinstance(payload, dict):
            LOGGER.warning("Model emitted non-object tool JSON: %s", raw[:500])
            continue
        function = payload.get("function", payload)
        if not isinstance(function, dict):
            LOGGER.warning("Model emitted invalid function payload: %s", raw[:500])
            continue
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
                encoded_arguments = arguments
            except json.JSONDecodeError:
                encoded_arguments = json.dumps({"input": arguments}, ensure_ascii=False)
        else:
            encoded_arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": encoded_arguments},
            }
        )
    remaining = TOOL_CALL_PATTERN.sub("", text).strip()
    return (remaining or None), calls
