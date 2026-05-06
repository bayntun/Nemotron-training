"""
Tests for eval.grader.

These pin down the exact behavior of the Kaggle scoring kernel so we catch
any drift introduced during refactors. If a test fails after editing
grader.py, you've changed semantics relative to the leaderboard grader -- do
not "fix" the test; revert grader.py instead.

Coverage:
- Boxed extraction (single, multiple, last-non-empty preference).
- Heuristic fallback patterns ("Final answer:", "the final answer is:", etc.).
- Last-numeric fallback.
- Last-non-empty-line fallback.
- NOT_FOUND for empty / None input.
- Numeric verify with math.isclose(rel_tol=1e-2, abs_tol=1e-5).
- String verify is case-insensitive AND whitespace-stripped on both sides.
- Mixed numeric vs non-numeric (graceful fall-through).
"""

from __future__ import annotations

from eval.grader import extract_final_answer, grade, verify

# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


class TestExtractBoxed:
    def test_simple_boxed(self):
        assert extract_final_answer(r"The answer is \boxed{42}") == "42"

    def test_boxed_with_text_around(self):
        text = r"After much thought, \boxed{XXXVIII} is correct."
        assert extract_final_answer(text) == "XXXVIII"

    def test_multiple_boxed_takes_last_nonempty(self):
        text = r"First guess \boxed{wrong}, but actually \boxed{right}."
        assert extract_final_answer(text) == "right"

    def test_empty_boxed_then_filled_returns_filled(self):
        text = r"Maybe \boxed{} or \boxed{42}."
        assert extract_final_answer(text) == "42"

    def test_only_empty_boxed_returns_empty(self):
        text = r"Confused: \boxed{}."
        assert extract_final_answer(text) == ""

    def test_boxed_with_whitespace_is_stripped(self):
        text = r"\boxed{  42  }"
        assert extract_final_answer(text) == "42"

    def test_boxed_unclosed_at_end(self):
        # The regex \\boxed\{([^}]*)(?:\}|$) accepts unclosed boxed at end.
        text = r"final: \boxed{42"
        assert extract_final_answer(text) == "42"


class TestExtractFallbackPatterns:
    def test_final_answer_is(self):
        text = "Working through... The final answer is: 7"
        assert extract_final_answer(text) == "7"

    def test_final_answer_colon(self):
        text = "Steps... Final answer: hello"
        assert extract_final_answer(text) == "hello"

    def test_final_answer_full_width_colon(self):
        text = "Steps... Final answer：hello"
        assert extract_final_answer(text) == "hello"

    def test_case_insensitive_pattern_match(self):
        text = "Working... FINAL ANSWER: 99"
        assert extract_final_answer(text) == "99"


class TestExtractNumericFallback:
    def test_last_number(self):
        text = "I tried 1, then 2, then 3, then I got 42."
        assert extract_final_answer(text) == "42"

    def test_last_negative_decimal(self):
        text = "Computed values: 3.14 then -2.718"
        assert extract_final_answer(text) == "-2.718"


class TestExtractLastLineFallback:
    def test_last_non_empty_line(self):
        text = "first line\n\nlast line\n\n"
        assert extract_final_answer(text) == "last line"


class TestExtractEdgeCases:
    def test_none_input(self):
        assert extract_final_answer(None) == "NOT_FOUND"

    def test_empty_string(self):
        assert extract_final_answer("") == "NOT_FOUND"

    def test_whitespace_only(self):
        assert extract_final_answer("   \n\t  ") == "NOT_FOUND"


# ---------------------------------------------------------------------------
# verify (numeric path)
# ---------------------------------------------------------------------------


class TestVerifyNumeric:
    def test_exact_int_match(self):
        assert verify("42", "42") is True

    def test_int_vs_float_string(self):
        assert verify("42", "42.0") is True

    def test_within_relative_tolerance(self):
        # 1% relative tolerance: 100 vs 100.5 -> 0.5% off, passes.
        assert verify("100", "100.5") is True

    def test_outside_relative_tolerance(self):
        # 1% relative: 100 vs 102 -> 2% off, fails.
        assert verify("100", "102") is False

    def test_at_relative_tolerance_boundary(self):
        # math.isclose with rel_tol=0.01 uses max(rel*abs(a), rel*abs(b), abs_tol).
        # 100 vs 101 -> 1% off, passes (math.isclose is inclusive).
        assert verify("100", "101") is True

    def test_absolute_tolerance_kicks_in_for_zero(self):
        # rel_tol=0 with both = 0 -> abs_tol=1e-5 saves us.
        assert verify("0", "0.000001") is True

    def test_negative_numbers(self):
        assert verify("-5", "-5.01") is True
        assert verify("-5", "-6") is False

    def test_scientific_notation(self):
        assert verify("1e3", "1000") is True


# ---------------------------------------------------------------------------
# verify (string fallback)
# ---------------------------------------------------------------------------


class TestVerifyString:
    def test_exact_string_match(self):
        assert verify("XXXVIII", "XXXVIII") is True

    def test_case_insensitive_match(self):
        assert verify("XXXVIII", "xxxviii") is True
        assert verify("Hello", "hello") is True

    def test_case_insensitive_mixed(self):
        assert verify("AbC", "aBc") is True

    def test_whitespace_stripped_on_both_sides(self):
        # verify() calls .strip() on BOTH stored_answer and predicted.
        assert verify("  hello  ", "hello") is True
        assert verify("hello", "  hello  ") is True
        assert verify("  hello  ", "  HELLO  ") is True

    def test_internal_whitespace_not_stripped(self):
        # Only leading/trailing strip; internal spaces are preserved.
        assert verify("hello world", "hello  world") is False

    def test_string_mismatch(self):
        assert verify("XXXVIII", "XXXVII") is False

    def test_one_numeric_one_string_fails_to_string_compare(self):
        # "42" parses; "forty-two" doesn't -> ValueError -> string compare.
        assert verify("42", "forty-two") is False

    def test_units_must_match_exactly(self):
        # The grader does NOT normalize units. "5 mph" != "5".
        assert verify("5 mph", "5") is False


# ---------------------------------------------------------------------------
# grade (end-to-end convenience wrapper)
# ---------------------------------------------------------------------------


class TestGradeEndToEnd:
    def test_correct_boxed_answer(self):
        text = r"<think>...</think>\boxed{42}"
        assert grade(text, "42") is True

    def test_correct_with_chain_of_thought(self):
        text = (
            "Let me work through this. First I notice the pattern is base 16."
            r" Converting 0xFF gives 255. Therefore: \boxed{255}"
        )
        assert grade(text, "255") is True

    def test_numeric_tolerance_within(self):
        text = r"After computation: \boxed{3.14}"
        assert grade(text, "3.14159") is True  # 0.05% off, passes

    def test_string_answer_with_case(self):
        text = r"\boxed{XXXVIII}"
        assert grade(text, "xxxviii") is True

    def test_wrong_answer(self):
        text = r"\boxed{wrong}"
        assert grade(text, "right") is False

    def test_no_boxed_falls_back_to_final_answer(self):
        text = "Working through... The final answer is: yes"
        assert grade(text, "yes") is True

    def test_completely_off_topic_grades_false(self):
        text = "This is a completely unrelated piece of text with no answer."
        # Last non-empty line fallback gives the whole text -> string mismatch.
        assert grade(text, "expected_answer") is False
