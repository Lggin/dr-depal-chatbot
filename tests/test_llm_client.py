"""LlamaCppClient logic tests that don't need a model."""

import pytest

from app.llm.client import LlamaCppClient


def test_strip_think_removes_leading_block():
    assert LlamaCppClient._strip_think("<think>\nblah\n</think>\n\n답변") == "답변"


def test_strip_think_keeps_text_without_block():
    assert LlamaCppClient._strip_think("  답변입니다  ") == "답변입니다"


def test_strip_think_unclosed_block_left_intact():
    assert LlamaCppClient._strip_think("<think>never closed") == "<think>never closed"


def test_methods_require_load():
    c = LlamaCppClient("nonexistent.gguf")
    with pytest.raises(RuntimeError):
        c.render([{"role": "user", "content": "x"}])
    with pytest.raises(RuntimeError):
        c.generate([{"role": "user", "content": "x"}])
    with pytest.raises(RuntimeError):
        c.count_tokens("x")
