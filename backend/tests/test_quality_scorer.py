"""Unit tests for quality scoring functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from backend.app.services.quality_scorer import score_page, score_document
from backend.app.models.schemas import PageIssue, IssueType, Severity


def _issue(severity: Severity) -> PageIssue:
    return PageIssue(
        type=IssueType.blank,
        severity=severity,
        description="test",
        location="full_page",
        confidence=0.9,
        recommended_action="rescan",
    )


class TestQualityScorer:
    def test_no_issues_gives_100(self):
        assert score_page([]) == 100.0

    def test_critical_issue_large_penalty(self):
        score = score_page([_issue(Severity.critical)])
        assert score == 75.0

    def test_warning_penalty(self):
        score = score_page([_issue(Severity.warning)])
        assert score == 90.0

    def test_multiple_issues_accumulate(self):
        issues = [_issue(Severity.critical), _issue(Severity.warning)]
        score = score_page(issues)
        assert score == 65.0

    def test_score_floors_at_zero(self):
        issues = [_issue(Severity.critical)] * 10
        score = score_page(issues)
        assert score == 0.0

    def test_document_score_uses_minimum_weighting(self):
        # One terrible page should drag the overall score down
        page_scores = [100.0, 100.0, 0.0]
        doc_score = score_document(page_scores)
        avg_score = score_document([100.0, 100.0, 100.0])
        assert doc_score < avg_score

    def test_all_perfect_pages_gives_100(self):
        assert score_document([100.0, 100.0, 100.0]) == 100.0
