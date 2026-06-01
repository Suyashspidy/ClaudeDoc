"""Computes overall and per-page quality scores from detected issues."""
from __future__ import annotations
from typing import List
from ..models.schemas import PageIssue, Severity

# Penalty points deducted from 100 per issue severity
SEVERITY_PENALTY: dict = {
    Severity.critical: 25.0,
    Severity.warning:  10.0,
    Severity.info:      3.0,
}


def score_page(issues: List[PageIssue]) -> float:
    """Return page quality score 0-100 based on detected issues."""
    if not issues:
        return 100.0
    penalty = sum(SEVERITY_PENALTY.get(i.severity, 0) for i in issues)
    return max(0.0, 100.0 - penalty)


def score_document(page_scores: List[float]) -> float:
    """Return overall document quality score as weighted average of page scores.

    Pages with critical issues drag the overall score down more aggressively
    via a quadratic penalty on the minimum score.
    """
    if not page_scores:
        return 100.0
    avg = sum(page_scores) / len(page_scores)
    min_score = min(page_scores)
    # Blend average with minimum to penalise worst pages
    return round(avg * 0.7 + min_score * 0.3, 2)
