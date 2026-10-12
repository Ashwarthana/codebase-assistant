"""
LLM generator abstraction layer supporting:
- Ollama (local LLaMA on CPU — default)
- Groq (cloud LLaMA via API)
- Google Gemini
- Anthropic Claude
- Mock (for testing without API keys)

All generators share a simple `generate(prompt) -> str` interface.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _retry(fn, max_retries: int = 2, delay: float = 1.0):
    """Simple retry wrapper for API calls."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                time.sleep(delay * (attempt + 1))
    raise last_error


class Generator(ABC):
    """Abstract base class for text generators."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response given a prompt."""
        pass


class OllamaGenerator(Generator):
    """Local LLM inference via Ollama (llama.cpp backend, runs on CPU)."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        self.base_url = (
            base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        )
        self.api_url = f"{self.base_url}/api/generate"

    def generate(self, prompt: str) -> str:
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 1024,
                "num_ctx": 8192,
            },
        }

        def _call():
            resp = requests.post(self.api_url, json=payload, timeout=240)
            resp.raise_for_status()
            return resp.json().get("response", "")

        try:
            return _retry(_call, max_retries=2)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return (
                "STATUS: INSUFFICIENT_EVIDENCE\n\n"
                "CITATIONS:\nNone\n\n"
                "ANSWER:\nFailed to generate answer due to Ollama connection error."
            )


class ClaudeGenerator(Generator):
    """Anthropic Claude API generator."""

    def __init__(self):
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy")
        )
        self.model = "claude-3-5-haiku-20241022"

    def generate(self, prompt: str) -> str:
        def _call():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return _retry(_call)


class GeminiGenerator(Generator):
    """Google Gemini API generator."""

    def __init__(self):
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "dummy"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt: str) -> str:
        import google.generativeai as genai

        def _call():
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
            return response.text

        return _retry(_call)


class GroqGenerator(Generator):
    """Groq cloud API generator (LLaMA 3.1 on LPU hardware)."""

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY", "dummy"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.1-8b-instant"

    def generate(self, prompt: str) -> str:
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        return _retry(_call)


class MockGenerator(Generator):
    """Mock generator for testing without any API keys."""

    def generate(self, prompt: str) -> str:
        return (
            "STATUS: SUPPORTED\n\n"
            "CITATIONS:\nNone\n\n"
            "ANSWER:\nThis is a mocked answer because no API key was provided."
        )


def get_generator() -> Generator:
    """Factory: create the configured generator based on GENERATOR env var.

    Supported values: ollama, groq, gemini, claude, mock
    Default: ollama (falls back to mock if Ollama is unreachable)
    """
    gen_type = os.environ.get("GENERATOR", "ollama").lower()

    if gen_type == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Using Claude generator")
        return ClaudeGenerator()
    elif gen_type == "gemini" and os.environ.get("GEMINI_API_KEY"):
        logger.info("Using Gemini generator")
        return GeminiGenerator()
    elif gen_type == "groq" and os.environ.get("GROQ_API_KEY"):
        logger.info("Using Groq generator")
        return GroqGenerator()
    elif gen_type == "ollama":
        logger.info("Using Ollama generator (local LLaMA)")
        return OllamaGenerator()
    else:
        logger.info("Using Mock generator")
        return MockGenerator()
