"""
TalentMatch AI - Intelligent Resume & Candidate Screening System
Machine Learning Task 3 (2026) | Future Interns Internship Program
Author: nithi-2008 | Repository: FUTURE_ML_03
"""

import json
import os
import sys
from typing import Dict, List
import pandas as pd
import plotly.express as px
import streamlit as st

# Add current workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.document_reader import reader
from src.explainer import explainer
from src.matcher import matcher
from src.ranker import ranker
from src.skill_extractor import extractor
from src.text_preprocessor import preprocessor
from src.visualizer import visualizer

# Set Page Configuration
st.set_page_config(
    page_title="TalentMatch AI | ML Resume Screening",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .badge-matched {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    .badge-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    .badge-pref {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    .badge-extra {
        background-color: #F3E8FF;
        color: #6B21A8;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_preloaded_job_descriptions() -> List[Dict]:
    """Loads standardized benchmark job descriptions."""
    jd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "job_descriptions.json")
    if os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_resumes_dataset() -> pd.DataFrame:
    """Loads the 120-candidate pre-generated dataset."""
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "resumes_dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_sample_resume_files() -> List[Dict[str, str]]:
    """Loads paths to sample candidate resumes."""
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_resumes")
    samples = []
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.endswith(".pdf") or f.endswith(".txt"):
                samples.append({
                    "filename": f,
                    "path": os.path.join(sample_dir, f),
                    "ext": os.path.splitext(f)[1].upper()
                })
    return samples


def main():
    # Header Section
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.markdown('<div class="main-header">💼 TalentMatch AI — Candidate Screening & Ranking</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Machine Learning Decision-Support System for Resume Screening, '
            'NLP Skill Extraction, Role Fit Ranking & Skill Gap Analysis. '
            '<b>Future Interns ML Task 3 (2026)</b> | Author: <b>nithi-2008</b></div>',
            unsafe_allow_html=True
        )
    with col_h2:
        st.link_button("🌐 GitHub Repository", "https://github.com/nithi-2008/FUTURE_ML_03.git", use_container_width=True)

    # Sidebar: Target Job Role & Configuration
    st.sidebar.header("🎯 Target Role & Job Description")

    jds = load_preloaded_job_descriptions()
    jd_options = [f"{jd['title']} ({jd['department']})" for jd in jds] + ["Custom Job Description..."]

    selected_jd_idx = st.sidebar.selectbox("Select Target Role Benchmark:", range(len(jd_options)), format_func=lambda i: jd_options[i])

    if selected_jd_idx < len(jds):
        active_jd = jds[selected_jd_idx]
        jd_title = active_jd["title"]
        jd_text = active_jd["description"]
        jd_required = active_jd["required_skills"]
        jd_preferred = active_jd["preferred_skills"]
        jd_exp = float(active_jd["experience_years"])
    else:
        st.sidebar.subheader("Custom Role Parameters")
        jd_title = st.sidebar.text_input("Job Title:", "Lead Software Engineer")
        jd_exp = st.sidebar.number_input("Required Years Experience:", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
        req_str = st.sidebar.text_area("Required Skills (comma-separated):", "Python, Docker, AWS, Git, SQL")
        pref_str = st.sidebar.text_area("Preferred Skills (comma-separated):", "Kubernetes, CI/CD, Redis, FastAPI")
        jd_text = st.sidebar.text_area("Job Description Details:", "We are looking for a skilled engineer to build scalable services...")
        jd_required = [s.strip() for s in req_str.split(",") if s.strip()]
        jd_preferred = [s.strip() for s in pref_str.split(",") if s.strip()]
        active_jd = {
            "title": jd_title,
            "description": jd_text,
            "required_skills": jd_required,
            "preferred_skills": jd_preferred,
            "experience_years": jd_exp
        }

    with st.sidebar.expander("⚙️ Scoring Model Weights", expanded=False):
        st.caption("Customize the decision-support algorithm weighting:")
        w_skill = st.slider("Skill Match Weight", 0.1, 0.8, 0.50, 0.05)
        w_semantic = st.slider("Semantic (TF-IDF) Weight", 0.1, 0.8, 0.35, 0.05)
        w_exp = st.slider("Experience Alignment Weight", 0.05, 0.5, 0.15, 0.05)
        weights_dict = {"skill": w_skill, "semantic": w_semantic, "experience": w_exp}

    with st.sidebar.expander("📌 Active Role Profile Summary", expanded=True):
        st.markdown(f"**Role**: `{jd_title}`")
        st.markdown(f"**Min Experience**: `{jd_exp} Years`")
        st.markdown(f"**Required Skills ({len(jd_required)})**:")
        st.write(", ".join([f"`{s}`" for s in jd_required]))
        st.markdown(f"**Preferred Skills ({len(jd_preferred)})**:")
        st.write(", ".join([f"`{s}`" for s in jd_preferred]))

    # Main Tabs
    tab_live, tab_inspect, tab_compare, tab_dataset, tab_sandbox = st.tabs([
        "📄 Live Resume Screening",
        "🔍 Candidate Deep-Dive & XAI",
        "⚖️ Side-by-Side Comparison",
        "🗄️ Batch Dataset Analytics",
        "📚 Skill Taxonomy Sandbox"
    ])

    # Initialize session state for evaluated results
    if "evaluated_candidates" not in st.session_state:
        st.session_state.evaluated_candidates = []

    # TAB 1: Live Resume Screening
    with tab_live:
        st.subheader("Candidate Resume Evaluation & Ranking")
        st.caption("Upload multiple resume files (PDF, DOCX, TXT) or load benchmark sample candidates.")

        c_up1, c_up2 = st.columns([3, 1])
        with c_up1:
            uploaded_files = st.file_uploader(
                "Upload Candidate Resumes (Drag & drop multiple files)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )
        with c_up2:
            st.markdown("<br>", unsafe_allow_html=True)
            load_samples_btn = st.button("📥 Load 6 Benchmark Resumes", use_container_width=True)

        candidate_eval_inputs = []

        if uploaded_files:
            for up_file in uploaded_files:
                file_bytes = up_file.read()
                try:
                    text, meta = reader.extract_text(file_bytes, filename=up_file.name)
                    if not meta["is_empty"]:
                        candidate_eval_inputs.append({"text": text, "filename": up_file.name})
                except Exception as e:
                    st.error(f"Error parsing {up_file.name}: {str(e)}")

        elif load_samples_btn or not st.session_state.evaluated_candidates:
            sample_files = load_sample_resume_files()
            # Only pick unique names (prefer PDF)
            seen_names = set()
            for s in sample_files:
                base = os.path.splitext(s["filename"])[0]
                if base not in seen_names and s["ext"] == ".PDF":
                    seen_names.add(base)
                    try:
                        text, meta = reader.extract_text(s["path"])
                        candidate_eval_inputs.append({"text": text, "filename": s["filename"]})
                    except Exception:
                        pass

        if candidate_eval_inputs:
            with st.spinner("Analyzing candidate text, extracting NLP skills, and computing similarity..."):
                raw_matches = []
                for item in candidate_eval_inputs:
                    match_res = matcher.match_resume(
                        resume_text=item["text"],
                        job_description=active_jd,
                        filename=item["filename"],
                        custom_weights=weights_dict
                    )
                    raw_matches.append(match_res)

                ranked_results = ranker.rank_candidates(raw_matches)
                st.session_state.evaluated_candidates = ranked_results

        # Display Results
        if st.session_state.evaluated_candidates:
            results = st.session_state.evaluated_candidates

            # Summary Metrics
            top_cand = results[0]
            high_count = sum(1 for c in results if "High" in c["tier"])
            avg_score = round(sum(c["composite_score"] for c in results) / len(results), 1)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Screened Resumes", len(results))
            m2.metric("Highly Recommended", f"{high_count} Candidates")
            m3.metric("Average Role Fit", f"{avg_score}%")
            m4.metric("Top Ranked", f"{top_cand['candidate_name']} ({top_cand['composite_score']}%)")

            st.markdown("---")

            # Leaderboard Table & Chart
            col_chart, col_tbl = st.columns([1, 1])

            with col_chart:
                fig_leaderboard = visualizer.plot_leaderboard(results, top_n=10)
                st.plotly_chart(fig_leaderboard, use_container_width=True)

            with col_tbl:
                st.markdown("##### 🏆 Candidate Ranking Leaderboard")
                table_rows = []
                for c in results:
                    table_rows.append({
                        "Rank": f"#{c['rank']}",
                        "Candidate Name": c["candidate_name"],
                        "Role Fit": f"{c['composite_score']}%",
                        "Tier": c["tier"],
                        "Experience": f"{c['experience_years']} yrs",
                        "Required Match": f"{len(c['matched_required'])}/{len(c['matched_required']) + len(c['missing_required'])}",
                        "Missing Req": ", ".join(c["missing_required"][:2]) if c["missing_required"] else "None"
                    })
                df_tbl = pd.DataFrame(table_rows)
                st.dataframe(df_tbl, use_container_width=True, hide_index=True)

                # Export CSV
                csv_data = pd.DataFrame(results).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Screening Report (CSV)",
                    data=csv_data,
                    file_name=f"screening_report_{jd_title.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        else:
            st.info("Upload resumes above or click 'Load 6 Benchmark Resumes' to start screening.")

    # TAB 2: Deep-Dive Candidate Inspector & XAI Explainer
    with tab_inspect:
        st.subheader("🔍 Explainable AI (XAI) Candidate Inspector")
        st.caption("Transparent mathematical score decomposition, skill gap matrix, and tailored interview probes.")

        if st.session_state.evaluated_candidates:
            candidates = st.session_state.evaluated_candidates
            cand_names = [f"#{c['rank']} {c['candidate_name']} ({c['composite_score']}%)" for c in candidates]
            sel_cand_idx = st.selectbox("Select Candidate to Inspect:", range(len(cand_names)), format_func=lambda i: cand_names[i])
            c_data = candidates[sel_cand_idx]

            # Top Header Info Card
            st.markdown(f"### {c_data['candidate_name']} — Rank #{c_data['rank']} ({c_data['tier']})")
            ci1, ci2, ci3, ci4 = st.columns(4)
            ci1.markdown(f"**📧 Email**: `{c_data['email']}`")
            ci2.markdown(f"**📞 Phone**: `{c_data['phone']}`")
            ci3.markdown(f"**⏳ Experience**: `{c_data['experience_years']} Years`")
            ci4.markdown(f"**🎓 Education**: `{c_data['highest_degree']}`")

            st.markdown(f"**Executive Verdict**: {c_data['verdict']}")

            # Score Breakdown Cards
            st.markdown("#### 📊 Score Decomposition")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Composite Role Fit", f"{c_data['composite_score']}%")
            sc2.metric("Skill Match Score", f"{c_data['skill_score']}%", f"{c_data['required_match_pct']}% req match")
            sc3.metric("Semantic Similarity", f"{c_data['semantic_score']}%", "TF-IDF Cosine")
            sc4.metric("Experience Alignment", f"{c_data['experience_score']}%", f"{c_data['experience_years']} yrs vs {jd_exp} yrs req")

            # Math explanation
            score_expl = explainer.explain_score_breakdown(c_data)
            st.info(f"💡 **Scoring Logic**: {score_expl['text_explanation']}")

            # Visualizations Row
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                # Skill gap donut
                fig_gap = visualizer.plot_skill_gap_summary(c_data)
                st.plotly_chart(fig_gap, use_container_width=True)

            with col_v2:
                # Domain Radar Chart
                jd_cats = {}
                for sk in active_jd.get("required_skills", []):
                    cat = extractor.canonical_to_category.get(sk, "General")
                    jd_cats[cat] = jd_cats.get(cat, 0) + 1
                fig_radar = visualizer.plot_radar_skills(c_data, jd_cats)
                st.plotly_chart(fig_radar, use_container_width=True)

            # Skill Matrix
            st.markdown("#### 🧩 Skill Coverage Matrix")
            sm1, sm2 = st.columns(2)

            with sm1:
                st.markdown("##### 🟢 Matched Required Skills:")
                if c_data["matched_required"]:
                    badges_html = " ".join([f'<span class="badge-matched">✓ {s}</span>' for s in c_data["matched_required"]])
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.write("None matched.")

                st.markdown("##### 🔵 Matched Preferred Skills:")
                if c_data["matched_preferred"]:
                    badges_html = " ".join([f'<span class="badge-pref">+ {s}</span>' for s in c_data["matched_preferred"]])
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.write("None matched.")

            with sm2:
                st.markdown("##### 🔴 Missing Required Skills (Critical Gaps):")
                if c_data["missing_required"]:
                    badges_html = " ".join([f'<span class="badge-missing">✗ {s}</span>' for s in c_data["missing_required"]])
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.success("None! 100% of required skills present.")

                st.markdown("##### 🟣 Additional Skills Detected Beyond JD:")
                if c_data["extra_skills"]:
                    badges_html = " ".join([f'<span class="badge-extra">• {s}</span>' for s in c_data["extra_skills"][:8]])
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.write("None detected.")

            # Smart Interview Probe Questions
            st.markdown("#### 🎯 Tailored Interview Verification Questions")
            questions = explainer.generate_interview_questions(c_data)
            if questions:
                for q in questions:
                    with st.expander(f"Question for **{q['skill']}** ({q['type']})"):
                        st.markdown(f"**Suggested Question**: *\"{q['question']}\"*")
                        st.caption(f"💡 **Recruiter Tip**: {q['recruiter_tip']}")
            else:
                st.write("Candidate covered all skills; standard behavioral interview recommended.")

            # Recruiter Briefing Report
            st.markdown("#### 📋 Executive Recruiter Briefing")
            report_text = explainer.generate_recruiter_report(c_data)
            st.code(report_text, language="markdown")

        else:
            st.info("No candidates evaluated yet. Please screen candidates in the 'Live Resume Screening' tab.")

    # TAB 3: Side-by-Side Candidate Comparison
    with tab_compare:
        st.subheader("⚖️ Head-to-Head Candidate Comparison")
        st.caption("Compare multiple candidates side-by-side to make confident hiring decisions.")

        if st.session_state.evaluated_candidates and len(st.session_state.evaluated_candidates) >= 2:
            candidates = st.session_state.evaluated_candidates
            options = {f"#{c['rank']} {c['candidate_name']} ({c['composite_score']}%)": c for c in candidates}

            selected_keys = st.multiselect(
                "Select 2 to 4 candidates to compare:",
                list(options.keys()),
                default=list(options.keys())[:2]
            )

            if len(selected_keys) >= 2:
                selected_cands = [options[k] for k in selected_keys]

                # Grouped Score Chart
                fig_comp = visualizer.plot_multi_comparison(selected_cands)
                st.plotly_chart(fig_comp, use_container_width=True)

                # Relative Advantage Breakdown
                st.markdown("#### 🔍 Comparative Analysis")
                c_lead = selected_cands[0]
                for c_other in selected_cands[1:]:
                    adv_text = explainer.explain_ranking_advantage(c_lead, c_other)
                    st.markdown(f"- {adv_text}")

                # Comparison Table
                st.markdown("#### 📊 Comparative Metric Matrix")
                cols = st.columns(len(selected_cands))
                for idx, c in enumerate(selected_cands):
                    with cols[idx]:
                        st.markdown(f"### #{c['rank']} {c['candidate_name']}")
                        st.metric("Composite Fit", f"{c['composite_score']}%", f"Rank #{c['rank']}")
                        st.write(f"**Tier**: {c['tier']}")
                        st.write(f"**Experience**: {c['experience_years']} yrs")
                        st.write(f"**Degree**: {c['highest_degree']}")
                        st.write(f"**Core Match**: {c['required_match_pct']}%")
                        st.markdown("**Matched Skills**:")
                        st.write(", ".join(c["matched_required"]) if c["matched_required"] else "None")
                        st.markdown("**Missing Required**:")
                        st.write(", ".join(c["missing_required"]) if c["missing_required"] else "None")
            else:
                st.warning("Please select at least 2 candidates to compare.")
        else:
            st.info("Need at least 2 evaluated candidates to compare. Please run screening first.")

    # TAB 4: Batch Dataset Analytics
    with tab_dataset:
        st.subheader("🗄️ Preloaded Resume Dataset & Batch Screening")
        st.caption("Screen the full pre-generated talent pool of 120 resumes across multiple domains.")

        df_dataset = load_resumes_dataset()

        if not df_dataset.empty:
            d1, d2, d3 = st.columns(3)
            d1.metric("Total Resumes in Dataset", len(df_dataset))
            d2.metric("Job Categories", df_dataset["primary_category"].nunique())
            d3.metric("Avg Candidate Experience", f"{round(df_dataset['experience_years'].mean(), 1)} yrs")

            if st.button("🚀 Screen All 120 Resumes Against Active Role", use_container_width=True):
                with st.spinner("Processing batch screening across 120 resumes..."):
                    batch_matches = []
                    for _, row in df_dataset.iterrows():
                        res = matcher.match_resume(
                            resume_text=row["resume_text"],
                            job_description=active_jd,
                            filename=f"{row['candidate_name'].lower().replace(' ', '_')}.txt",
                            custom_weights=weights_dict
                        )
                        res["primary_category"] = row["primary_category"]
                        batch_matches.append(res)

                    ranked_batch = ranker.rank_candidates(batch_matches)
                    st.session_state.batch_results = ranked_batch

            if "batch_results" in st.session_state:
                b_results = st.session_state.batch_results
                st.success(f"Successfully screened {len(b_results)} candidates against '{jd_title}'!")

                # Distribution of scores by category
                df_b = pd.DataFrame(b_results)
                fig_box = px.box(
                    df_b,
                    x="primary_category",
                    y="composite_score",
                    color="primary_category",
                    title="<b>Score Distribution by Candidate Background Domain</b>",
                    labels={"primary_category": "Domain", "composite_score": "Composite Fit (%)"}
                )
                fig_box.update_layout(xaxis_tickangle=-45, showlegend=False, height=400)
                st.plotly_chart(fig_box, use_container_width=True)

                # Filterable batch table
                st.markdown("##### 📋 Filter Batch Results")
                f1, f2 = st.columns(2)
                with f1:
                    sel_cat = st.multiselect("Filter by Domain:", df_b["primary_category"].unique(), default=list(df_b["primary_category"].unique())[:3])
                with f2:
                    min_score_filter = st.slider("Minimum Composite Score:", 0, 100, 50)

                filtered_df = df_b[(df_b["primary_category"].isin(sel_cat)) & (df_b["composite_score"] >= min_score_filter)]
                display_cols = ["rank", "candidate_name", "primary_category", "composite_score", "skill_score", "semantic_score", "tier"]
                st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

        else:
            st.warning("Dataset file not found.")

    # TAB 5: Skill Taxonomy & NLP Sandbox
    with tab_sandbox:
        st.subheader("📚 Skill Taxonomy & NLP Matcher Sandbox")
        st.caption("Inspect the 500+ skills taxonomy and test real-time NLP extraction on custom text.")

        sample_sandbox_text = st.text_area(
            "Paste any resume paragraph, job posting, or candidate bio:",
            "I have 6 years of experience building machine learning pipelines using Python, PyTorch, Scikit-Learn, "
            "and Hugging Face transformers. Deployed FastAPI services containerized with Docker on AWS EKS with Kubernetes. "
            "Managed PostgreSQL databases and automated CI/CD with GitHub Actions.",
            height=140
        )

        if st.button("🔍 Extract Skills with NLP", use_container_width=True):
            nlp_res = extractor.extract_skills(sample_sandbox_text)
            st.success(f"Extracted {nlp_res['total_skills_count']} Unique Canonical Skills!")

            s_cols = st.columns(2)
            with s_cols[0]:
                st.markdown("##### 📌 Detected Canonical Skills:")
                st.write(nlp_res["found_skills"])

            with s_cols[1]:
                st.markdown("##### 🏷️ Skills Grouped by Category:")
                for cat, sks in nlp_res["skills_by_category"].items():
                    st.markdown(f"**{cat}**: {', '.join(sks)}")


if __name__ == "__main__":
    main()
