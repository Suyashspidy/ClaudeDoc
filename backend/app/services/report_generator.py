"""AI-powered QA report generator using Google Gemini API."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ..models.schemas import QAReport, PageReport, Severity

logger = logging.getLogger(__name__)


_REPORT_SYSTEM_PROMPT = """You are a senior document quality assurance specialist.
Generate professional, plain-English QA reports for scanned document analysis results.
Be concise but thorough. Focus on actionable recommendations.
Structure your response as a professional assessment, not a technical dump."""

_REPORT_HUMAN_TEMPLATE = """Analyse this document QA scan result and write a professional summary report:

Document: {document_name}
Total Pages: {total_pages}
Overall Quality Score: {quality_score:.1f}/100
Processing Time: {processing_time:.1f}s

Issues Found:
- Critical: {critical_count}
- Warnings: {warning_count}
- Info: {info_count}

Scanner Health:
- Artifacts Detected: {artifact_detected}
- Pages Affected by Artifacts: {artifact_pages}
- Cleaning Recommended: {cleaning_recommended}

Page-level Details:
{page_details}

Write a 3-4 paragraph professional QA report including:
1. Executive summary of document quality
2. Key issues found and their business impact
3. Recommended remediation actions in priority order
4. Overall disposition (Accept / Accept with Caveats / Reject for Rescan)
"""


class AIReportGenerator:
    """Generates natural language QA narratives using Google Gemini."""

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            from config.settings import settings
            genai.configure(api_key=settings.google_api_key)
            self._model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                system_instruction=_REPORT_SYSTEM_PROMPT,
            )
        return self._model

    def generate_narrative(self, report: QAReport) -> str:
        """Return a professional plain-English narrative for the QA report."""
        page_details = self._format_page_details(report.pages)
        prompt = _REPORT_HUMAN_TEMPLATE.format(
            document_name=report.document_name,
            total_pages=report.total_pages,
            quality_score=report.overall_quality_score,
            processing_time=report.processing_time_seconds,
            critical_count=report.issues_summary.critical,
            warning_count=report.issues_summary.warnings,
            info_count=report.issues_summary.info,
            artifact_detected=report.scanner_health.artifact_detected,
            artifact_pages=report.scanner_health.affected_pages,
            cleaning_recommended=report.scanner_health.cleaning_recommended,
            page_details=page_details,
        )

        try:
            model = self._get_model()
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            return self._fallback_narrative(report)

    def _format_page_details(self, pages: List[PageReport]) -> str:
        lines = []
        for page in pages:
            if page.issues:
                issue_strs = [f"{i.type.value}({i.severity.value})" for i in page.issues]
                lines.append(f"  Page {page.page_number}: {', '.join(issue_strs)} — score {page.quality_score:.0f}/100")
        return "\n".join(lines) if lines else "  No issues detected on any page."

    def _fallback_narrative(self, report: QAReport) -> str:
        score = report.overall_quality_score
        disposition = "Accept" if score >= 85 else "Accept with Caveats" if score >= 60 else "Reject for Rescan"
        return (
            f"Document '{report.document_name}' received an overall quality score of "
            f"{score:.1f}/100. "
            f"Analysis identified {report.issues_summary.critical} critical issues, "
            f"{report.issues_summary.warnings} warnings, and {report.issues_summary.info} informational notices "
            f"across {report.total_pages} pages. "
            f"Disposition: {disposition}."
        )


class PDFReportBuilder:
    """Builds a downloadable PDF QA report with thumbnails and issue tables."""

    def build(self, report: QAReport, thumbnail_dir: Path, output_path: Path) -> Path:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import cm
        except ImportError:
            logger.warning("reportlab not installed, skipping PDF report generation")
            return output_path

        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle("title", parent=styles["Heading1"], fontSize=18)
        story.append(Paragraph(f"Document QA Report: {report.document_name}", title_style))
        story.append(Spacer(1, 0.3 * cm))

        # Summary table
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        summary_data = [
            ["Metric", "Value"],
            ["Overall Quality Score", f"{report.overall_quality_score:.1f}/100"],
            ["Total Pages", str(report.total_pages)],
            ["Critical Issues", str(report.issues_summary.critical)],
            ["Warnings", str(report.issues_summary.warnings)],
            ["Processing Time", f"{report.processing_time_seconds:.1f}s"],
            ["Scanner Cleaning Needed", "Yes" if report.scanner_health.cleaning_recommended else "No"],
        ]
        table = Table(summary_data, colWidths=[8 * cm, 8 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.5 * cm))

        # AI Narrative
        if report.ai_narrative:
            story.append(Paragraph("AI Assessment", styles["Heading2"]))
            story.append(Paragraph(report.ai_narrative, styles["Normal"]))
            story.append(Spacer(1, 0.5 * cm))

        # Per-page issues
        story.append(Paragraph("Page-Level Issues", styles["Heading2"]))
        for page in report.pages:
            if not page.issues:
                continue
            story.append(Paragraph(f"Page {page.page_number} — Score: {page.quality_score:.0f}/100", styles["Heading3"]))

            thumb_path = thumbnail_dir / f"page_{page.page_number}_annotated.jpg"
            if thumb_path.exists():
                story.append(Image(str(thumb_path), width=8 * cm, height=11 * cm))

            for issue in page.issues:
                color_map = {Severity.critical: "#E74C3C", Severity.warning: "#F39C12", Severity.info: "#3498DB"}
                badge_color = color_map.get(issue.severity, "#999999")
                story.append(Paragraph(
                    f'<font color="{badge_color}"><b>[{issue.severity.value.upper()}]</b></font> '
                    f'{issue.type.value}: {issue.description}',
                    styles["Normal"]
                ))
                story.append(Paragraph(f"<i>Action: {issue.recommended_action}</i>", styles["Italic"]))
                story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph(
            f"Report generated by ClaudeDoc QA Agent — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"]
        ))

        doc.build(story)
        return output_path
