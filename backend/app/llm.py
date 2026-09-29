"""LLM generation layer.

Thin wrapper around the existing Gemini client that handles errors
cleanly and keeps API-key management in one place.  This module has a
single responsibility: send a prompt string and return the generated
text.

Reuses the ``genai.Client()`` already initialised in ``services.py``
so there is exactly one client instance across the application.
"""

from __future__ import annotations

from app.config import LLM_MODEL
from app.services import client as gemini_client


def generate_answer(prompt: str) -> str:
    """Sends *prompt* to the configured Gemini model and returns the text.

    Args:
        prompt: The fully-constructed prompt (context + question already
                merged by the caller).

    Returns:
        The model's response text.

    Raises:
        ValueError:  If *prompt* is empty.
        RuntimeError: If the Gemini API call fails or returns an empty
                      response.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    try:
        response = gemini_client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
        )
    except Exception as exc:
        raise RuntimeError(
            f"LLM generation failed: {exc}"
        ) from exc

    # Gemini can return a response object whose .text is None or empty
    # when the model declines to answer or hits a safety filter.
    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise RuntimeError(
            "LLM returned an empty response. The model may have declined "
            "to answer or hit a safety filter."
        )

    return text.strip()
