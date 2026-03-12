"""
Switchable LLM provider supporting Anthropic Claude, OpenAI GPT,
Ollama (local), and llama.cpp (local GGUF models).
Provides a unified interface for the agent to call any backend.
"""

import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from config.settings import settings, LLMProvider

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM."""
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Generate a response with tool/function calling support."""
        ...


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""

    def __init__(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when using the Anthropic provider")

        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model
        logger.info(f"Initialized Anthropic client with model: {self.model}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or settings.llm_max_tokens,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens or settings.llm_max_tokens,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        # Convert tools to Anthropic tool format
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
            )

        message = await self.client.messages.create(
            model=self.model,
            max_tokens=settings.llm_max_tokens,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            system=system_prompt or "",
            tools=anthropic_tools,
            messages=[{"role": "user", "content": prompt}],
        )

        result = {"text": "", "tool_calls": []}
        for block in message.content:
            if block.type == "text":
                result["text"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append(
                    {"name": block.name, "arguments": block.input, "id": block.id}
                )
        return result


class OpenAIClient(LLMClient):
    """OpenAI GPT client."""

    def __init__(self):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using the OpenAI provider")

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        logger.info(f"Initialized OpenAI client with model: {self.model}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
        return response.choices[0].message.content

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Convert to OpenAI function calling format
        openai_tools = []
        for tool in tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            tools=openai_tools,
        )

        result = {"text": response.choices[0].message.content or "", "tool_calls": []}
        if response.choices[0].message.tool_calls:
            import json

            for tc in response.choices[0].message.tool_calls:
                result["tool_calls"].append(
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                        "id": tc.id,
                    }
                )
        return result


class OllamaClient(LLMClient):
    """
    Ollama local LLM client.
    Connects to a running Ollama server (ollama serve) via its REST API.
    Supports models like qwen3, llama3, mistral, phi3, etc.
    """

    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        logger.info(f"Initialized Ollama client: {self.base_url} model={self.model}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.llm_temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,  # local models can be slow
            )
            resp.raise_for_status()
            data = resp.json()

        return data.get("message", {}).get("content", "")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        import httpx
        import json as _json

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else settings.llm_temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        chunk = _json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Ollama tool calling — supported in newer Ollama versions (0.4+)
        for models that support function calling (qwen3, llama3.1+, mistral).
        Falls back to prompt-based tool selection for older models.
        """
        import httpx
        import json as _json

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Convert to Ollama tool format (OpenAI-compatible)
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            })

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "tools": ollama_tools,
            "options": {
                "temperature": temperature if temperature is not None else settings.llm_temperature,
            },
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()

            result = {"text": "", "tool_calls": []}
            message = data.get("message", {})
            result["text"] = message.get("content", "")

            # Parse Ollama tool calls
            for tc in message.get("tool_calls", []):
                func = tc.get("function", {})
                result["tool_calls"].append({
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", {}),
                    "id": f"ollama_{func.get('name', '')}",
                })

            return result

        except Exception as e:
            # Fallback: if tool calling fails, use prompt-based approach
            logger.warning(f"Ollama tool calling failed ({e}), falling back to prompt-based")
            tool_descriptions = "\n".join(
                f"- {t['name']}: {t['description']}" for t in tools
            )
            enhanced_prompt = (
                f"{prompt}\n\nAvailable tools:\n{tool_descriptions}\n\n"
                f"If you need to use a tool, respond with the tool name and arguments. "
                f"Otherwise, answer directly."
            )
            text = await self.generate(enhanced_prompt, system_prompt, temperature)
            return {"text": text, "tool_calls": []}


class LlamaCppClient(LLMClient):
    """
    llama.cpp client supporting both:
    1. llama-server (HTTP API) — for running models via llama.cpp's built-in server
    2. llama-cpp-python — for in-process model loading via Python bindings

    Configure via LLAMACPP_BASE_URL for server mode,
    or LLAMACPP_MODEL_PATH for in-process mode.
    """

    def __init__(self):
        self.base_url = settings.llamacpp_base_url.rstrip("/")
        self.model_path = settings.llamacpp_model_path
        self._local_model = None

        if self.model_path:
            # In-process mode via llama-cpp-python
            try:
                from llama_cpp import Llama
                self._local_model = Llama(
                    model_path=self.model_path,
                    n_ctx=settings.llamacpp_n_ctx,
                    n_gpu_layers=settings.llamacpp_n_gpu_layers,
                    verbose=False,
                )
                logger.info(f"Loaded GGUF model: {self.model_path}")
            except ImportError:
                raise ImportError(
                    "Install llama-cpp-python: pip install llama-cpp-python"
                )
        else:
            logger.info(f"Using llama.cpp server at: {self.base_url}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self._local_model:
            return await self._generate_local(prompt, system_prompt, temperature, max_tokens)
        return await self._generate_server(prompt, system_prompt, temperature, max_tokens)

    async def _generate_local(
        self, prompt, system_prompt, temperature, max_tokens
    ) -> str:
        """Generate using llama-cpp-python (in-process)."""
        import asyncio

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Run in thread pool since llama-cpp-python is synchronous
        def _run():
            response = self._local_model.create_chat_completion(
                messages=messages,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
            return response["choices"][0]["message"]["content"]

        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def _generate_server(
        self, prompt, system_prompt, temperature, max_tokens
    ) -> str:
        """Generate using llama-server HTTP API."""
        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "n_predict": max_tokens or settings.llm_max_tokens,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            # llama-server uses /v1/chat/completions (OpenAI-compatible)
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        import httpx
        import json as _json

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self._local_model:
            # Stream from local model
            import asyncio

            def _stream():
                return self._local_model.create_chat_completion(
                    messages=messages,
                    temperature=temperature if temperature is not None else settings.llm_temperature,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    stream=True,
                )

            stream = await asyncio.get_event_loop().run_in_executor(None, _stream)
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
        else:
            # Stream from llama-server
            payload = {
                "messages": messages,
                "temperature": temperature if temperature is not None else settings.llm_temperature,
                "n_predict": max_tokens or settings.llm_max_tokens,
                "stream": True,
            }

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            chunk = _json.loads(line[6:])
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Tool calling via prompt engineering for llama.cpp models.
        Most GGUF models don't natively support function calling,
        so we embed tool descriptions in the prompt.
        """
        tool_descriptions = "\n".join(
            f"- **{t['name']}**: {t['description']}\n"
            f"  Parameters: {_json_compact(t['parameters'])}"
            for t in tools
        )

        enhanced_prompt = f"""{prompt}

You have access to the following tools. To use a tool, respond with a JSON block:
```json
{{"tool": "tool_name", "arguments": {{...}}}}
```

Available tools:
{tool_descriptions}

If you can answer directly without tools, just respond normally."""

        text = await self.generate(enhanced_prompt, system_prompt, temperature)

        # Try to parse tool calls from the response
        result = {"text": text, "tool_calls": []}
        import re
        import json as _json

        json_blocks = re.findall(r'```json\s*(\{[^`]+\})\s*```', text)
        for block in json_blocks:
            try:
                parsed = _json.loads(block)
                if "tool" in parsed:
                    result["tool_calls"].append({
                        "name": parsed["tool"],
                        "arguments": parsed.get("arguments", {}),
                        "id": f"llamacpp_{parsed['tool']}",
                    })
                    # Remove the JSON block from the text response
                    result["text"] = text.replace(f"```json\n{block}\n```", "").strip()
            except _json.JSONDecodeError:
                continue

        return result


def _json_compact(obj) -> str:
    """Compact JSON for prompt inclusion."""
    import json
    return json.dumps(obj, separators=(",", ":"))[:200]


def get_llm_client(provider: Optional[LLMProvider] = None) -> LLMClient:
    """Factory function to get the configured LLM client."""
    provider = provider or settings.llm_provider
    if provider == LLMProvider.ANTHROPIC:
        return AnthropicClient()
    elif provider == LLMProvider.OPENAI:
        return OpenAIClient()
    elif provider == LLMProvider.OLLAMA:
        return OllamaClient()
    elif provider == LLMProvider.LLAMACPP:
        return LlamaCppClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
