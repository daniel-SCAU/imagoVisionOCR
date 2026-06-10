"""
Validation engine.

Supports:
  * exact match (case-insensitive by default)
  * regex match
  * multi-field / multi-rule validation

Rules config format (stored as a JSON string in the database or inline dict):

  {
    "lot_number": "LOT240615A",
    "expiry":     "\\d{4}-\\d{2}-\\d{2}"
  }

For a single extracted string use :func:`validate_text`.
For field-level validation use :func:`validate_fields`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    reason: str
    details: dict[str, bool]

    def __str__(self) -> str:
        return "PASS" if self.passed else f"FAIL – {self.reason}"


def _load_rules(rules_input: Any) -> dict[str, str]:
    """Accept rules as a dict, JSON string, or empty value."""
    if isinstance(rules_input, dict):
        return rules_input
    if isinstance(rules_input, str) and rules_input.strip():
        try:
            loaded = json.loads(rules_input)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse validation rules JSON: %s", exc)
    return {}


def _is_regex(pattern: str) -> bool:
    """Heuristic: if the pattern contains regex meta-chars it is treated as regex."""
    return bool(re.search(r"[\\^$.*+?()[\]{}|]", pattern))


def validate_text(
    text: str,
    rules: Any,
    case_sensitive: bool = False,
) -> ValidationResult:
    """
    Validate a single extracted OCR string against all rules.

    The text must satisfy **every** rule to PASS.

    Args:
        text:           The cleaned OCR output.
        rules:          Dict or JSON string mapping field name → expected
                        value / regex pattern.
        case_sensitive: If False (default), comparisons are case-insensitive.

    Returns:
        :class:`ValidationResult` with passed flag, reason, and per-rule detail.
    """
    rule_dict = _load_rules(rules)
    if not rule_dict:
        return ValidationResult(passed=True, reason="No rules configured", details={})

    compare_text = text if case_sensitive else text.upper()
    details: dict[str, bool] = {}
    failures: list[str] = []

    for field, pattern in rule_dict.items():
        compare_pattern = pattern if case_sensitive else pattern.upper()
        if _is_regex(compare_pattern):
            matched = bool(re.search(compare_pattern, compare_text))
        else:
            matched = compare_text == compare_pattern
        details[field] = matched
        if not matched:
            failures.append(f"{field!r}: expected {pattern!r}, got {text!r}")

    passed = len(failures) == 0
    reason = "; ".join(failures) if failures else "All rules matched"
    return ValidationResult(passed=passed, reason=reason, details=details)


def validate_fields(
    fields: dict[str, str],
    rules: Any,
    case_sensitive: bool = False,
) -> ValidationResult:
    """
    Validate a dictionary of extracted fields against corresponding rules.

    Args:
        fields: Mapping of field name → extracted text (e.g. from multi-ROI OCR).
        rules:  Mapping of field name → expected value / regex pattern.
        case_sensitive: Case handling.

    Returns:
        :class:`ValidationResult` with per-field detail.
    """
    rule_dict = _load_rules(rules)
    if not rule_dict:
        return ValidationResult(passed=True, reason="No rules configured", details={})

    details: dict[str, bool] = {}
    failures: list[str] = []

    for field, pattern in rule_dict.items():
        extracted = fields.get(field, "")
        compare_extracted = extracted if case_sensitive else extracted.upper()
        compare_pattern = pattern if case_sensitive else pattern.upper()

        if _is_regex(compare_pattern):
            matched = bool(re.search(compare_pattern, compare_extracted))
        else:
            matched = compare_extracted == compare_pattern

        details[field] = matched
        if not matched:
            failures.append(
                f"{field!r}: expected {pattern!r}, got {extracted!r}"
            )

    passed = len(failures) == 0
    reason = "; ".join(failures) if failures else "All rules matched"
    return ValidationResult(passed=passed, reason=reason, details=details)
