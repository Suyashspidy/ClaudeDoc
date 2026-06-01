"""Streamlit frontend for the ClaudeDoc QA Agent."""
from __future__ import annotations
import json
import time

import requests
import streamlit as st

API_BASE = "http://165.245.143.135:8000/api/v1"

st.set_page_config(
    page_title="ClaudeDoc — Book Scan QA",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ClaudeDoc")
    st.caption("AI Book Scan Quality Assurance")
    st.divider()
    st.markdown("**What this agent detects:**")
    st.markdown("""
- 📄 Missing pages
- 📐 Skewed / rotated pages
- ⬜ Blank pages
- 🔲 Folded corners
- 🖨️ Scanner artifacts
- 🚫 Foreign objects on page
""")
    st.divider()
    st.markdown("**Your workflow:**")
    st.markdown("""
1. Upload scanned PDF
2. See which pages have issues
3. Rescan only those pages
4. Deliver clean PDF to client
""")


# ── Helper: render report ────────────────────────────────────────────────────
def _render_report(report: dict) -> None:
    pages_with_issues = [p for p in report["pages"] if p["issues"]]
    total = report["total_pages"]
    problem_count = len(pages_with_issues)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pages", total)
    col2.metric("Pages to Rescan", problem_count,
                delta=f"{problem_count} need attention" if problem_count else None,
                delta_color="inverse")
    col3.metric("Critical Issues", report["issues_summary"]["critical"])
    col4.metric("Warnings", report["issues_summary"]["warnings"])

    if problem_count == 0:
        st.success("No issues found. This scan is clean and ready to send to your client.")
        return

    # ── Rescan table ──
    st.subheader(f"Pages to Rescan ({problem_count})")
    st.markdown("Rescan each of these pages and replace them in your PDF before delivery.")

    SEVERITY_COLOR = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
    ISSUE_LABEL = {
        "missing_page":     "MISSING PAGE",
        "blank_page":       "BLANK PAGE",
        "folded_page":      "FOLDED CORNER",
        "scanner_artifact": "SCANNER ARTIFACT",
        "foreign_object":   "FOREIGN OBJECT",
        "skew":             "SKEWED / ROTATED",
    }

    rows = []
    for page in sorted(pages_with_issues, key=lambda p: p["page_number"]):
        for issue in page["issues"]:
            icon = SEVERITY_COLOR.get(issue["severity"], "⚪")
            label = ISSUE_LABEL.get(issue["type"], issue["type"].upper())
            rows.append({
                "Page": page["page_number"],
                "Severity": f"{icon} {issue['severity'].upper()}",
                "Issue": label,
                "Description": issue["description"],
                "Action": issue["recommended_action"],
            })

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Page":        st.column_config.NumberColumn(width="small"),
            "Severity":    st.column_config.TextColumn(width="medium"),
            "Issue":       st.column_config.TextColumn(width="medium"),
            "Description": st.column_config.TextColumn(width="large"),
            "Action":      st.column_config.TextColumn(width="large"),
        }
    )

    # ── Quick copy list ──
    page_nums = sorted(set(p["page_number"] for p in pages_with_issues))
    st.subheader("Pages to Rescan (copy-paste list)")
    st.code(", ".join(str(p) for p in page_nums), language=None)

    # ── Scanner health ──
    health = report["scanner_health"]
    if health["cleaning_recommended"]:
        st.warning(
            f"**Scanner cleaning recommended.** Artifacts found on "
            f"{len(health['affected_pages'])} pages at consistent positions — "
            f"this is a dirty scanner lens. Clean the glass before rescanning."
        )

    # ── Download ──
    st.divider()
    st.download_button(
        "Download Full Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"qa_report_{report['job_id'][:8]}.json",
        mime="application/json",
    )


# ── Main upload page ─────────────────────────────────────────────────────────
st.title("📖 Book Scan QA")
st.markdown("Upload a scanned PDF to find pages that need to be rescanned before sending to your client.")

uploaded_file = st.file_uploader(
    "Drop your scanned PDF here",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
    help="Max 200 MB.",
)

if uploaded_file is not None:
    st.info(f"**{uploaded_file.name}** — {uploaded_file.size / 1024 / 1024:.1f} MB")

    if st.button("Run QA Check", type="primary", use_container_width=True):

        with st.spinner("Uploading…"):
            resp = requests.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=120,
            )
        if resp.status_code != 200:
            st.error(f"Upload failed: {resp.text}")
            st.stop()

        job_id = resp.json()["job_id"]

        progress_bar = st.progress(0, text="Starting…")
        while True:
            sr = requests.get(f"{API_BASE}/status/{job_id}", timeout=10)
            if sr.status_code != 200:
                st.error("Lost connection to server.")
                st.stop()
            s = sr.json()
            pct = int(s.get("progress", 0))
            msg = s.get("message", "Processing…")
            progress_bar.progress(min(pct, 100), text=msg)

            if s["status"] == "completed":
                break
            if s["status"] == "failed":
                st.error(f"Processing failed: {s.get('error', 'Unknown error')}")
                st.stop()
            time.sleep(2)

        progress_bar.progress(100, text="Done!")

        rr = requests.get(f"{API_BASE}/report/{job_id}", timeout=30)
        if rr.status_code != 200:
            st.error("Could not load report.")
            st.stop()

        _render_report(rr.json())
