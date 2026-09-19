"""Unit tests for build_rag_prompt — the RAG prompt-builder function.

Tests verify the prompt's structural properties (contains question,
context, citation instructions, no-answer fallback) without calling the LLM.
"""

import pytest

from app.services import build_rag_prompt, NO_ANSWER_MESSAGE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_QUESTION = "What parser does the AST chunking system use?"
SAMPLE_CONTEXT = (
    '[Source 1 — File: "report.pdf", Page 1, Chunk 0]\n'
    "The system uses Python's ast module via ast.NodeVisitor.\n\n"
    '[Source 2 — File: "report.pdf", Page 2, Chunk 1]\n'
    "It targets FunctionDef and ClassDef nodes."
)


class TestBuildRagPromptStructure:
    """Verify the prompt contains the required structural elements."""

    def test_contains_question(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        assert SAMPLE_QUESTION in prompt

    def test_contains_context(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        assert SAMPLE_CONTEXT in prompt

    def test_context_appears_before_question(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        ctx_pos = prompt.index(SAMPLE_CONTEXT)
        q_pos = prompt.index(SAMPLE_QUESTION)
        assert ctx_pos < q_pos, "Context must appear before the question"

    def test_ends_with_answer_prompt(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        assert prompt.rstrip().endswith("Answer:")


class TestBuildRagPromptGrounding:
    """Verify the prompt enforces strict grounding to context only."""

    def test_instructs_answer_only_from_context(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        prompt_lower = prompt.lower()
        assert "only" in prompt_lower
        assert "context" in prompt_lower

    def test_forbids_outside_knowledge(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        prompt_lower = prompt.lower()
        assert "outside knowledge" in prompt_lower

    def test_forbids_speculation(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        prompt_lower = prompt.lower()
        assert "speculate" in prompt_lower


class TestBuildRagPromptInsufficientContext:
    """Verify the prompt handles the case where context is insufficient."""

    def test_contains_no_answer_fallback_message(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        assert NO_ANSWER_MESSAGE in prompt

    def test_instructs_exact_fallback_response(self):
        """The prompt must tell the LLM to respond with *exactly* the fallback."""
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        prompt_lower = prompt.lower()
        assert "exactly" in prompt_lower

    def test_fallback_with_empty_context(self):
        """Even with empty context the prompt still includes the fallback."""
        prompt = build_rag_prompt(SAMPLE_QUESTION, "")
        assert NO_ANSWER_MESSAGE in prompt


class TestBuildRagPromptCitations:
    """Verify the prompt requires source citations and forbids fabrication."""

    def test_instructs_to_cite_sources(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        prompt_lower = prompt.lower()
        assert "cite" in prompt_lower

    def test_references_source_label_format(self):
        """The prompt should mention the [Source N] label format."""
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        assert "[Source 1]" in prompt or "[Source" in prompt

    def test_forbids_fabricated_citations(self):
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        prompt_lower = prompt.lower()
        assert "fabricat" in prompt_lower  # matches "fabricate" / "fabricated"

    def test_restricts_citations_to_provided_labels(self):
        """The prompt must say to only cite labels that actually appear."""
        prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        prompt_lower = prompt.lower()
        # Must reference limiting citations to what's in the context
        assert "only cite" in prompt_lower or "only reference" in prompt_lower


class TestBuildRagPromptReturnType:
    """Basic contract checks."""

    def test_returns_string(self):
        result = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        assert isinstance(result, str)

    def test_non_empty(self):
        result = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CONTEXT)
        assert len(result) > 0

    def test_different_questions_produce_different_prompts(self):
        p1 = build_rag_prompt("What is X?", SAMPLE_CONTEXT)
        p2 = build_rag_prompt("What is Y?", SAMPLE_CONTEXT)
        assert p1 != p2

    def test_different_contexts_produce_different_prompts(self):
        p1 = build_rag_prompt(SAMPLE_QUESTION, "Context A")
        p2 = build_rag_prompt(SAMPLE_QUESTION, "Context B")
        assert p1 != p2
