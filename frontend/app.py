"""Streamlit recruiter dashboard for the AI-powered ATS."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Final

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL: Final[str] = "http://127.0.0.1:8000"
REQUEST_TIMEOUT: Final[int] = 120
SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".pdf", ".docx")

st.set_page_config(
    page_title="AI-Powered ATS - Recruiter Intelligence",
    layout="wide",
    page_icon="🎯",
)

st.markdown(
    """
    <style>
      :root { --ink: #e9f0ff; --muted: #91a2be; --panel: #101b2e; --line: #22324b; --accent: #61d5c2; }
      .stApp { background: radial-gradient(ellipse at 80% 0%, #172a43 0, #0a1220 48%, #080e19 100%); color: var(--ink); }
      html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, sans-serif; }
      h1, h2, h3 { font-family: ui-sans-serif, system-ui, sans-serif; letter-spacing: -0.035em; }
      h1 { color: #f2f6ff; font-weight: 800; }
      [data-testid="stSidebar"] { background: #0b1422; border-right: 1px solid var(--line); }
      [data-testid="stMetric"] { background: linear-gradient(145deg, #14233a, #101a2b); border: 1px solid #263952; padding: 1.05rem 1.2rem; border-radius: 14px; }
      [data-testid="stMetricLabel"] { color: #a9bad3; }
      [data-testid="stMetricValue"] { color: #eff6ff; }
      [data-testid="stVerticalBlockBorderWrapper"] { border-color: #263952 !important; }
      .hero { padding: 1.4rem 1.6rem; border-radius: 18px; border: 1px solid #2b4059; background: linear-gradient(115deg, rgba(29,52,77,.88), rgba(16,27,46,.86)); margin: 0.3rem 0 1.5rem; }
      .eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; margin-bottom: .45rem; }
      .hero-copy { color: #a9bad3; margin: .3rem 0 0; font-size: 1rem; }
      .skill-badge { display: inline-block; padding: .42rem .75rem; margin: .2rem .35rem .2rem 0; border-radius: 999px; font-size: .86rem; font-weight: 600; }
      .skill-good { color: #a4f2d7; background: #103d38; border: 1px solid #287d6b; }
      .skill-missing { color: #ffc1bb; background: #482628; border: 1px solid #8a4142; }
      .small-note { color: #91a2be; font-size: .9rem; }
      div.stButton > button[kind="primary"] { background: linear-gradient(105deg, #56cdbb, #43aebd); border: none; color: #071722; font-weight: 700; min-height: 2.8rem; }
      div.stButton > button[kind="primary"]:hover { background: linear-gradient(105deg, #77e4d0, #61c8d4); color: #071722; border: none; }
      div[data-testid="stExpander"] { background: rgba(16,27,46,.7); border: 1px solid #263952; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _request_error_message(error: requests.RequestException) -> str:
    if isinstance(error, requests.Timeout):
        return "The ATS service took too long to respond. Please try again in a moment."
    if isinstance(error, requests.ConnectionError):
        return "Could not connect to the ATS backend. Confirm that it is running at http://127.0.0.1:8000."
    return f"Could not complete the request: {error}"


def _read_response(response: requests.Response) -> dict[str, Any] | None:
    try:
        response.raise_for_status()
    except requests.HTTPError:
        try:
            error_body: Any = response.json()
            detail: Any = (
                error_body.get("detail", response.text)
                if isinstance(error_body, dict)
                else response.text
            )
        except ValueError:
            detail = response.text or "The backend returned an error."
        st.error(f"Backend error ({response.status_code}): {detail}")
        return None

    try:
        result: Any = response.json()
    except ValueError:
        st.error("The backend returned an unreadable response. Please try again.")
        return None
    if not isinstance(result, dict):
        st.error("The backend returned a response in an unexpected format.")
        return None
    return result


def _show_skills(title: str, skills: list[str], css_class: str, empty_message: str) -> None:
    with st.expander(title, expanded=True):
        if not skills:
            st.caption(empty_message)
            return
        badges: str = "".join(
            f'<span class="skill-badge {css_class}">{html.escape(skill)}</span>'
            for skill in skills
        )
        st.markdown(badges, unsafe_allow_html=True)


def _blind_contact_details(contact: dict[str, Any]) -> dict[str, str]:
    raw_details: dict[str, str] = {
        "name": str(contact.get("name") or ""),
        "email": str(contact.get("email") or ""),
        "phone": str(contact.get("phone") or ""),
    }
    if not any(raw_details.values()):
        return {"name": "Not detected", "email": "Not detected", "phone": "Not detected"}

    try:
        response: requests.Response = requests.post(
            f"{API_BASE_URL}/api/anonymize-text",
            json={"text": "\n".join(raw_details.values())},
            timeout=REQUEST_TIMEOUT,
        )
        result: dict[str, Any] | None = _read_response(response)
    except requests.RequestException as error:
        st.warning(f"Blind review masking could not reach the backend: {_request_error_message(error)}")
        return {
            key: "[REDACTED]" if value else "Not detected"
            for key, value in raw_details.items()
        }
    if result is None:
        return {
            key: "[REDACTED]" if value else "Not detected"
            for key, value in raw_details.items()
        }

    anonymized_text: str = str(result.get("anonymized_text", ""))
    placeholders: tuple[str, ...] = (
        "[REDACTED_NAME]",
        "[REDACTED_EMAIL]",
        "[REDACTED_PHONE]",
    )
    output: dict[str, str] = {}
    for index, (key, value) in enumerate(raw_details.items()):
        if not value:
            output[key] = "Not detected"
            continue
        placeholder: str = placeholders[index]
        output[key] = (
            placeholder
            if placeholder in anonymized_text
            else "[REDACTED]"
        )
    if any(value in anonymized_text for value in raw_details.values() if value):
        return {
            key: "[REDACTED]" if value else "Not detected"
            for key, value in raw_details.items()
        }
    if not any(placeholder in anonymized_text for placeholder in placeholders):
        return {
            key: "[REDACTED]" if value else "Not detected"
            for key, value in raw_details.items()
        }
    return output


def _render_single_results(result: dict[str, Any], blind_review: bool) -> None:
    overall_score: float = float(result.get("overall_score", 0.0))
    semantic_score: float = float(result.get("semantic_score", 0.0)) * 100
    skill_score: float = float(result.get("skill_match_percentage", 0.0))
    bm25_score: float = float(result.get("bm25_score", 0.0)) * 100

    st.markdown("### Candidate fit")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Overall Match Score", f"{overall_score:.1f}%")
    metric_columns[1].metric("Semantic Similarity", f"{semantic_score:.1f}%")
    metric_columns[2].metric("Hard Skill Match", f"{skill_score:.1f}%")

    chart_data: pd.DataFrame = pd.DataFrame(
        {
            "Signal": ["Semantic similarity", "Hard skill match", "Keyword relevance"],
            "Score": [semantic_score, skill_score, bm25_score],
        }
    )
    chart = px.bar(
        chart_data,
        x="Score",
        y="Signal",
        orientation="h",
        range_x=[0, 100],
        color="Score",
        color_continuous_scale=["#425779", "#61d5c2"],
        text=chart_data["Score"].map(lambda score: f"{score:.1f}%"),
    )
    chart.update_layout(
        height=250,
        margin=dict(l=12, r=18, t=12, b=12),
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce8f8"),
        xaxis=dict(showgrid=True, gridcolor="#263952", ticksuffix="%"),
        yaxis=dict(title=None, autorange="reversed"),
    )
    chart.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(chart, use_container_width=True)

    matched_skills: list[str] = result.get("matched_skills", [])
    missing_skills: list[str] = result.get("missing_skills", [])
    skill_columns = st.columns(2)
    with skill_columns[0]:
        _show_skills("Matched Skills", matched_skills, "skill-good", "No required skills matched.")
    with skill_columns[1]:
        _show_skills("Missing Skills", missing_skills, "skill-missing", "No missing required skills.")

    contact: dict[str, Any] = result.get("contact_details", {})
    if blind_review:
        contact_display: dict[str, str] = _blind_contact_details(contact)
    else:
        contact_display = {
            key: str(contact.get(key) or "Not detected")
            for key in ("name", "email", "phone")
        }
    st.markdown("### Candidate details")
    detail_columns = st.columns(4)
    detail_columns[0].metric("Name", contact_display["name"])
    detail_columns[1].metric("Email", contact_display["email"])
    detail_columns[2].metric("Phone", contact_display["phone"])
    detail_columns[3].metric(
        "Estimated Experience",
        f"{float(result.get('detected_experience_years', 0.0)):.1f} years",
    )
    if blind_review:
        st.caption("Blind review is on: personal contact details are masked.")


def _render_single_screener() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Precision screening</div>
          <h1 style="margin:0">Candidate Screener</h1>
          <p class="hero-copy">Compare a resume with the role and see the signals behind each match.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    resume = st.file_uploader("Resume", type=["pdf", "docx"], key="single_resume")
    job_description: str = st.text_area(
        "Job Description",
        height=190,
        placeholder="Paste the role description and responsibilities...",
        key="single_job_description",
    )
    required_skills: str = st.text_input(
        "Required Skills",
        placeholder="Python, Docker, PyTorch",
        help="Enter the essential skills separated by commas.",
        key="single_required_skills",
    )
    blind_review: bool = st.checkbox(
        "Enable Anonymized / Blind Review",
        value=False,
        key="single_blind_review",
    )

    if st.button("Screen Candidate", type="primary", use_container_width=True):
        st.session_state.pop("single_screen_result", None)
        if resume is None:
            st.error("Upload a PDF or DOCX resume to continue.")
        elif Path(resume.name).suffix.casefold() not in SUPPORTED_EXTENSIONS:
            st.error("Unsupported file format. Please upload a PDF or DOCX resume.")
        elif not job_description.strip():
            st.error("Add a job description before screening the candidate.")
        else:
            with st.spinner("Parsing resume and calculating match signals..."):
                try:
                    response: requests.Response = requests.post(
                        f"{API_BASE_URL}/api/screen-single",
                        files={
                            "resume": (
                                resume.name,
                                resume.getvalue(),
                                resume.type or "application/octet-stream",
                            )
                        },
                        data={
                            "job_description": job_description,
                            "required_skills": required_skills,
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    result: dict[str, Any] | None = _read_response(response)
                except requests.RequestException as error:
                    st.error(_request_error_message(error))
                    result = None
            if result is not None:
                st.session_state["single_screen_result"] = result

    saved_result: Any = st.session_state.get("single_screen_result")
    if isinstance(saved_result, dict):
        _render_single_results(saved_result, blind_review)


def _render_bulk_dashboard() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Search your talent pool</div>
          <h1 style="margin:0">Talent Pool &amp; Ranking</h1>
          <p class="hero-copy">Index resumes once, then surface the strongest candidates for each new role.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 01 · Build your talent pool")
    uploaded_files = st.file_uploader(
        "Resume files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="bulk_resumes",
    )
    if st.button("Index Resumes into Vector DB", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Choose one or more PDF or DOCX resumes to index.")
        else:
            invalid_files: list[str] = [
                resume.name
                for resume in uploaded_files
                if Path(resume.name).suffix.casefold() not in SUPPORTED_EXTENSIONS
            ]
            if invalid_files:
                st.error(f"Unsupported file format: {', '.join(invalid_files)}. Use PDF or DOCX files.")
            else:
                multipart_files: list[tuple[str, tuple[str, bytes, str]]] = [
                    (
                        "files",
                        (
                            resume.name,
                            resume.getvalue(),
                            resume.type or "application/octet-stream",
                        ),
                    )
                    for resume in uploaded_files
                ]
                with st.spinner(f"Indexing {len(multipart_files)} resume(s)..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/upload-bulk",
                            files=multipart_files,
                            timeout=REQUEST_TIMEOUT,
                        )
                        result = _read_response(response)
                    except requests.RequestException as error:
                        st.error(_request_error_message(error))
                        result = None
                if result is not None:
                    indexed_count: int = int(result.get("indexed_count", 0))
                    st.success(f"Indexed {indexed_count} candidate(s) into the talent pool.")
                    candidates: Any = result.get("candidates", [])
                    if isinstance(candidates, list) and candidates:
                        st.dataframe(
                            pd.DataFrame(candidates),
                            use_container_width=True,
                            hide_index=True,
                        )

    st.divider()
    st.markdown("### 02 · Rank candidates for a role")
    job_description: str = st.text_area(
        "Job Description",
        height=180,
        placeholder="Paste the job description to rank your indexed talent pool...",
        key="pool_job_description",
    )
    if st.button("Rank Entire Candidate Pool", type="primary", use_container_width=True):
        if not job_description.strip():
            st.error("Add a job description before ranking candidates.")
        else:
            try:
                health_response: requests.Response = requests.get(
                    f"{API_BASE_URL}/health",
                    timeout=15,
                )
                health_result: dict[str, Any] | None = _read_response(health_response)
            except requests.RequestException as error:
                st.error(_request_error_message(error))
                health_result = None

            if health_result is not None:
                indexed_resumes: int = int(health_result.get("indexed_resumes", 0))
                if indexed_resumes == 0:
                    st.info("Your talent pool is empty. Index resumes before ranking candidates.")
                else:
                    with st.spinner("Searching and scoring the full indexed pool..."):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/api/rank-pool",
                                json={
                                    "job_description": job_description,
                                    "required_skills": [],
                                    "top_k": indexed_resumes,
                                },
                                timeout=REQUEST_TIMEOUT,
                            )
                            result = _read_response(response)
                        except requests.RequestException as error:
                            st.error(_request_error_message(error))
                            result = None
                    if result is not None:
                        leaderboard: Any = result.get("leaderboard", [])
                        if not isinstance(leaderboard, list):
                            st.error("The backend returned an unexpected leaderboard format.")
                        elif not leaderboard:
                            st.info("No candidates were returned for this search.")
                        else:
                            leaderboard.sort(
                                key=lambda candidate: float(candidate.get("overall_score", 0.0)),
                                reverse=True,
                            )
                            st.session_state["candidate_leaderboard"] = leaderboard

    saved_leaderboard: Any = st.session_state.get("candidate_leaderboard")
    if isinstance(saved_leaderboard, list) and saved_leaderboard:
        table_rows: list[dict[str, Any]] = [
            {
                "Candidate ID / Name": (
                    f"{candidate.get('candidate_name') or 'Unknown'} · "
                    f"{candidate.get('candidate_id', '')}"
                ),
                "Overall Match %": round(float(candidate.get("overall_score", 0.0)), 1),
                "Key Extracted Skills": ", ".join(candidate.get("skills", [])),
            }
            for candidate in saved_leaderboard
        ]
        st.markdown("### Ranked candidates")
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Overall Match %": st.column_config.ProgressColumn(
                    "Overall Match %",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                )
            },
        )


st.sidebar.markdown("## ATS Recruiter")
st.sidebar.caption("Recruiter intelligence workspace")
navigation: str = st.sidebar.radio(
    "Workspace",
    ("Candidate Screener (Single)", "Talent Pool & Ranking (Bulk)"),
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="small-note">Connected service<br><b>127.0.0.1:8000</b></span>',
    unsafe_allow_html=True,
)

if navigation == "Candidate Screener (Single)":
    _render_single_screener()
else:
    _render_bulk_dashboard()
