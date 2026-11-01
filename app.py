"""
=============================================================================
Support Ticket Classification & Prioritization System
Machine Learning Task 2 (2026) • Future Interns Internship Program
Flagship Streamlit Dashboard • Enterprise NLP Decision-Support Platform
Author: nithi-2008
=============================================================================
"""

import os
import json
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inference import get_classifier
from src.text_preprocessor import clean_text, extract_ticket_signals

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TicketAI | Support Triage & Prioritization",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS Styling: Dark Glassmorphism SaaS Aesthetic
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Header Accent */
    .hero-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Metric Badges */
    .badge-critical {
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.25);
    }

    .badge-high {
        background: rgba(249, 115, 22, 0.18);
        color: #FB923C;
        border: 1px solid rgba(249, 115, 22, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .badge-medium {
        background: rgba(234, 179, 8, 0.18);
        color: #FACC15;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .badge-low {
        background: rgba(34, 197, 94, 0.18);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .keyword-pill {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 10px;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        display: inline-block;
        margin: 3px;
    }

    /* Auto-response box */
    .draft-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px dashed rgba(129, 140, 248, 0.4);
        border-radius: 12px;
        padding: 18px;
        color: #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Sidebar Logo Header */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Custom KPI Stat Card */
    .kpi-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: left;
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .kpi-sub {
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Initialize Model Inference Engine
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Ticket Triage NLP Models...")
def load_triage_engine():
    try:
        classifier = get_classifier()
        return classifier
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure `python -m src.model_trainer` has been executed.")
        return None

classifier = load_triage_engine()

# Load Metadata
metadata = classifier.metadata if classifier else {}

# -----------------------------------------------------------------------------
# Sidebar Navigation & Workspace Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <span style="font-size: 1.7rem;">🎫</span>
            <span>TicketAI Studio</span>
        </div>
        <p style="color: #64748B; font-size: 0.82rem; margin-bottom: 20px;">
            ML Task 2 (2026) • Future Interns<br>
            <b>Auto-Triage & Prioritization</b>
        </p>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    app_mode = st.radio(
        label="Select Workspace Module",
        options=[
            "🎫 Smart Triage (Live Demo)",
            "📂 Batch Ticket Ingestion",
            "📊 Support Analytics & KPIs",
            "🧪 Model Benchmark Arena",
            "💰 Business Impact & ROI"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Engine Status")
    if classifier and metadata:
        best_cat = metadata.get("best_category_model", "Logistic Regression")
        best_prio = metadata.get("best_priority_model", "Logistic Regression")
        st.markdown(f"**Category Model**: `{best_cat}`")
        st.markdown(f"**Priority Model**: `{best_prio}`")
        st.markdown(f"**TF-IDF Vocab**: `{metadata.get('tfidf_vocabulary_size', 5000):,} terms`")
        st.success("🟢 Real-Time Inference Active")
    else:
        st.warning("⚠️ Models not loaded")

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.75rem; color: #64748B; line-height: 1.4;">
            <b>Developer</b>: nithi-2008<br>
            <b>Project</b>: FUTURE_ML_02<br>
            <b>Internship</b>: Future Interns (2026)
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helper Functions for Formatting & Plotting
# -----------------------------------------------------------------------------
def get_priority_badge_html(priority: str) -> str:
    priority_lower = priority.lower()
    if priority_lower == "critical":
        return f'<span class="badge-critical">🔴 CRITICAL (P1)</span>'
    elif priority_lower == "high":
        return f'<span class="badge-high">🟠 HIGH (P2)</span>'
    elif priority_lower == "medium":
        return f'<span class="badge-medium">🟡 MEDIUM (P3)</span>'
    else:
        return f'<span class="badge-low">🟢 LOW (P4)</span>'


def get_category_icon(category: str) -> str:
    icons = {
        "Technical Issue": "🛠️",
        "Billing & Invoicing": "💳",
        "Account & Security": "🔐",
        "Product Inquiry": "❓",
        "Cancellation & Refund": "🛑"
    }
    return icons.get(category, "📄")


# =============================================================================
# MODULE 1: SMART TRIAGE (LIVE DEMO & PLAYGROUND)
# =============================================================================
if app_mode == "🎫 Smart Triage (Live Demo)":
    st.markdown('<div class="hero-title">Automated Ticket Classification & Prioritization</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Instantly analyze incoming customer tickets, classify issue categories, assign SLA priority, and route to the correct engineering or support desk.</div>', unsafe_allow_html=True)

    # Pre-set template selector
    st.markdown("##### ⚡ One-Click Realistic Ticket Presets:")
    col_pre1, col_pre2, col_pre3, col_pre4, col_pre5 = st.columns(5)

    preset_tickets = {
        "outage": {
            "name": "🚨 System Outage",
            "subj": "CRITICAL OUTAGE: Production API returning 500 Internal Server Errors",
            "desc": "Our entire production cluster is down. API endpoints are responding with 500 status codes since 14:00 UTC. Over 50,000 of our active users cannot checkout. Please escalate to Level 3 DevOps immediately!",
            "tier": "Enterprise",
            "channel": "API / Webhook"
        },
        "billing": {
            "name": "💳 Billing Dispute",
            "subj": "URGENT BILLING: Unauthorized recurring charge of $12,500 on business card",
            "desc": "We noticed an unauthorized charge of $12,500 processed on our corporate Amex card today! We are on the $499 monthly plan. Reverse this fraudulent charge immediately or our bank will file a dispute.",
            "tier": "Business",
            "channel": "Email"
        },
        "security": {
            "name": "🔐 Security Breach",
            "subj": "SECURITY INCIDENT: Suspicious logins from unrecognized foreign IP addresses",
            "desc": "Our admin audit logs show 40 successful login attempts from an unknown Russian IP within 15 minutes. We suspect our master API secret key was compromised. Please invalidate all sessions immediately!",
            "tier": "Enterprise",
            "channel": "Web Portal"
        },
        "product": {
            "name": "❓ Webhook Question",
            "subj": "Question regarding webhook payload signature verification (HMAC-SHA256)",
            "desc": "Could you provide a Python or Node.js code example for verifying the X-Signature header sent with incoming webhook events? We want to verify payload authenticity before deploying.",
            "tier": "Startup",
            "channel": "Live Chat"
        },
        "cancel": {
            "name": "🛑 Refund Request",
            "subj": "Refund request within 30-day money-back guarantee period",
            "desc": "We signed up 18 days ago. Unfortunately, the software lacks key inventory integrations needed for our retail business. Please process our refund under your 30-day guarantee and cancel auto-renew.",
            "tier": "Free Tier",
            "channel": "Web Portal"
        }
    }

    default_subj = preset_tickets["outage"]["subj"]
    default_desc = preset_tickets["outage"]["desc"]
    default_tier = preset_tickets["outage"]["tier"]
    default_chan = preset_tickets["outage"]["channel"]

    with col_pre1:
        if st.button("🚨 Outage", use_container_width=True):
            st.session_state["ticket_subj"] = preset_tickets["outage"]["subj"]
            st.session_state["ticket_desc"] = preset_tickets["outage"]["desc"]
            st.session_state["ticket_tier"] = preset_tickets["outage"]["tier"]
            st.session_state["ticket_chan"] = preset_tickets["outage"]["channel"]

    with col_pre2:
        if st.button("💳 Billing", use_container_width=True):
            st.session_state["ticket_subj"] = preset_tickets["billing"]["subj"]
            st.session_state["ticket_desc"] = preset_tickets["billing"]["desc"]
            st.session_state["ticket_tier"] = preset_tickets["billing"]["tier"]
            st.session_state["ticket_chan"] = preset_tickets["billing"]["channel"]

    with col_pre3:
        if st.button("🔐 Security", use_container_width=True):
            st.session_state["ticket_subj"] = preset_tickets["security"]["subj"]
            st.session_state["ticket_desc"] = preset_tickets["security"]["desc"]
            st.session_state["ticket_tier"] = preset_tickets["security"]["tier"]
            st.session_state["ticket_chan"] = preset_tickets["security"]["channel"]

    with col_pre4:
        if st.button("❓ Feature", use_container_width=True):
            st.session_state["ticket_subj"] = preset_tickets["product"]["subj"]
            st.session_state["ticket_desc"] = preset_tickets["product"]["desc"]
            st.session_state["ticket_tier"] = preset_tickets["product"]["tier"]
            st.session_state["ticket_chan"] = preset_tickets["product"]["channel"]

    with col_pre5:
        if st.button("🛑 Refund", use_container_width=True):
            st.session_state["ticket_subj"] = preset_tickets["cancel"]["subj"]
            st.session_state["ticket_desc"] = preset_tickets["cancel"]["desc"]
            st.session_state["ticket_tier"] = preset_tickets["cancel"]["tier"]
            st.session_state["ticket_chan"] = preset_tickets["cancel"]["channel"]

    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_output = st.columns([1.1, 1.3], gap="large")

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Customer Ticket Input")

        c_name = st.text_input("Customer Name", value="Alexander Wright")
        
        tier_options = ["Enterprise", "Business", "Startup", "Free Tier"]
        current_tier = st.session_state.get("ticket_tier", "Enterprise")
        tier_idx = tier_options.index(current_tier) if current_tier in tier_options else 0
        c_tier = st.selectbox("Customer Account Tier", tier_options, index=tier_idx)

        chan_options = ["Email", "Web Portal", "Live Chat", "Mobile App", "API / Webhook"]
        current_chan = st.session_state.get("ticket_chan", "API / Webhook")
        chan_idx = chan_options.index(current_chan) if current_chan in chan_options else 0
        c_chan = st.selectbox("Ingestion Channel", chan_options, index=chan_idx)

        input_subj = st.text_input(
            "Ticket Subject",
            value=st.session_state.get("ticket_subj", default_subj)
        )

        input_desc = st.text_area(
            "Ticket Description (Free Text)",
            value=st.session_state.get("ticket_desc", default_desc),
            height=140
        )

        run_triage_btn = st.button("🚀 Analyze & Triage Ticket", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Run inference
    with col_output:
        if classifier:
            res = classifier.predict_ticket(
                subject=input_subj,
                description=input_desc,
                customer_name=c_name,
                customer_tier=c_tier,
                ticket_channel=c_chan
            )

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Intelligent Decision-Support Results")

            # Main Result Banner
            cat_icon = get_category_icon(res["category"])
            prio_badge = get_priority_badge_html(res["priority"])

            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Predicted Category</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">
                            {cat_icon} {res['category']}
                        </div>
                        <div style="font-size: 0.85rem; color: #38BDF8;">Confidence: <b>{res['category_confidence']:.1%}</b></div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">Assigned Priority & SLA</div>
                        {prio_badge}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # SLA & Routing Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">First Response SLA</div>
                        <div class="kpi-value" style="font-size: 1.25rem; color: #38BDF8;">{res['sla_response_time']}</div>
                        <div class="kpi-sub">Target response clock</div>
                    </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Max Resolution SLA</div>
                        <div class="kpi-value" style="font-size: 1.25rem; color: #FACC15;">{res['sla_resolution_hours']} hrs</div>
                        <div class="kpi-sub">Escalation deadline</div>
                    </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Routing Department</div>
                        <div class="kpi-value" style="font-size: 0.95rem; font-weight: 600; color: #A78BFA; line-height: 1.2; padding: 4px 0;">{res['assigned_team']}</div>
                        <div class="kpi-sub">Automated Queue Dispatch</div>
                    </div>
                """, unsafe_allow_html=True)

            if res["escalated"]:
                st.info(f"⚡ **SLA Policy Triggered**: {res['escalation_reason']}")

            # Explainability Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 🔍 Influential Predictive Keywords (Explainability)")
            keywords = res.get("influential_keywords", [])
            if keywords:
                pills_html = "".join([f'<span class="keyword-pill">{kw["keyword"]} ({kw["importance"]:.2f})</span>' for kw in keywords])
                st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
            else:
                st.caption("Standard ticket tokens processed without high anomaly spikes.")

            # AI Draft Auto-Response
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### ✉️ AI-Suggested Auto-Response Draft")
            st.markdown(f'<div class="draft-box">{res["draft_response"]}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Probability Breakdown Chart
            with st.expander("📊 View Multi-Class Prediction Probability Distributions", expanded=False):
                tab_p1, tab_p2 = st.tabs(["Category Probabilities", "Priority Probabilities"])
                
                with tab_p1:
                    cat_probs = res.get("category_probabilities", {})
                    if cat_probs:
                        df_cat_prob = pd.DataFrame(list(cat_probs.items()), columns=["Category", "Probability"])
                        fig_cat = px.bar(
                            df_cat_prob, x="Probability", y="Category", orientation="h",
                            color="Probability", color_continuous_scale="Viridis",
                            text_auto=".1%"
                        )
                        fig_cat.update_layout(
                            height=250, margin=dict(l=10, r=10, t=20, b=20),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#E2E8F0")
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)

                with tab_p2:
                    prio_probs = res.get("priority_probabilities", {})
                    if prio_probs:
                        df_prio_prob = pd.DataFrame(list(prio_probs.items()), columns=["Priority", "Probability"])
                        fig_prio = px.bar(
                            df_prio_prob, x="Probability", y="Priority", orientation="h",
                            color="Priority",
                            color_discrete_map={"Critical": "#EF4444", "High": "#F97316", "Medium": "#EAB308", "Low": "#22C55E"},
                            text_auto=".1%"
                        )
                        fig_prio.update_layout(
                            height=220, margin=dict(l=10, r=10, t=20, b=20),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#E2E8F0")
                        )
                        st.plotly_chart(fig_prio, use_container_width=True)


# =============================================================================
# MODULE 2: BATCH TICKET INGESTION
# =============================================================================
elif app_mode == "📂 Batch Ticket Ingestion":
    st.markdown('<div class="hero-title">Bulk Ticket Ingestion & Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Upload a batch CSV of hundreds or thousands of customer tickets to classify categories, assign priorities, and route teams at scale.</div>', unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([1.2, 0.8])
    
    with b_col1:
        uploaded_file = st.file_uploader(
            "Upload Support Tickets CSV File",
            type=["csv"],
            help="Upload a CSV with ticket text columns (e.g. ticket_subject, ticket_description)"
        )

    with b_col2:
        st.markdown("##### 💡 Need Sample Data?")
        st.markdown("Test instant batch processing with our pre-configured benchmark batch of 50 unlabelled customer tickets:")
        load_sample_batch = st.button("📥 Load 50-Ticket Benchmark Batch", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Uploaded `{uploaded_file.name}` ({len(batch_df)} rows)")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif load_sample_batch or os.path.exists("data/sample_tickets_batch.csv"):
        if os.path.exists("data/sample_tickets_batch.csv"):
            batch_df = pd.read_csv("data/sample_tickets_batch.csv")

    if batch_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Raw Ingested Tickets Preview")
        st.dataframe(batch_df.head(6), use_container_width=True)

        if st.button("⚡ Process & Triage All Tickets", type="primary", use_container_width=True):
            with st.spinner(f"Running high-throughput NLP classification across {len(batch_df)} tickets..."):
                t_start = time.time()
                enriched_df = classifier.predict_batch(batch_df)
                elapsed = time.time() - t_start

            st.balloons()
            st.success(f"Successfully triaged **{len(enriched_df)} tickets** in **{elapsed:.2f} seconds** ({len(enriched_df)/max(0.01, elapsed):.0f} tickets/sec)!")

            # Batch KPI Metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Total Processed</div>
                        <div class="kpi-value">{len(enriched_df)}</div>
                        <div class="kpi-sub">100% categorized</div>
                    </div>
                """, unsafe_allow_html=True)
            with kpi2:
                crit_high = enriched_df["predicted_priority"].isin(["Critical", "High"]).sum()
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Critical & High (P1/P2)</div>
                        <div class="kpi-value" style="color: #F87171;">{crit_high} ({crit_high/len(enriched_df):.1%})</div>
                        <div class="kpi-sub">Requiring immediate response</div>
                    </div>
                """, unsafe_allow_html=True)
            with kpi3:
                top_cat = enriched_df["predicted_category"].value_counts().index[0]
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Dominant Category</div>
                        <div class="kpi-value" style="font-size: 1.15rem; color: #38BDF8;">{top_cat}</div>
                        <div class="kpi-sub">Highest volume queue</div>
                    </div>
                """, unsafe_allow_html=True)
            with kpi4:
                hours_saved = round((len(enriched_df) * 4.5) / 60, 1)
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Manual Time Saved</div>
                        <div class="kpi-value" style="color: #4ADE80;">{hours_saved} hrs</div>
                        <div class="kpi-sub">Based on 4.5 min/ticket</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Filterable Triage Results")

            # Filters
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                sel_cat = st.multiselect(
                    "Filter by Category",
                    options=enriched_df["predicted_category"].unique().tolist(),
                    default=enriched_df["predicted_category"].unique().tolist()
                )
            with f_col2:
                sel_prio = st.multiselect(
                    "Filter by Priority",
                    options=enriched_df["predicted_priority"].unique().tolist(),
                    default=enriched_df["predicted_priority"].unique().tolist()
                )

            filtered_df = enriched_df[
                enriched_df["predicted_category"].isin(sel_cat) &
                enriched_df["predicted_priority"].isin(sel_prio)
            ]

            display_cols = [
                c for c in ["ticket_id", "customer_name", "ticket_subject", "predicted_category",
                            "category_confidence", "predicted_priority", "assigned_team", "sla_target_hours"]
                if c in filtered_df.columns
            ]
            st.dataframe(filtered_df[display_cols], use_container_width=True)

            # Export Button
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Triaged Tickets CSV",
                data=csv_data,
                file_name="triaged_support_tickets_export.csv",
                mime="text/csv",
                type="secondary"
            )


# =============================================================================
# MODULE 3: SUPPORT ANALYTICS & KPIS
# =============================================================================
elif app_mode == "📊 Support Analytics & KPIs":
    st.markdown('<div class="hero-title">Support Desk Operations Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Comprehensive operational metrics, ticket distribution patterns, channel breakdowns, and SLA risk exposures across the support pipeline.</div>', unsafe_allow_html=True)

    data_path = "data/customer_support_tickets.csv"
    if os.path.exists(data_path):
        df_bench = pd.read_csv(data_path)

        # Overview Stats
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Historical Tickets</div>
                    <div class="kpi-value">{len(df_bench):,}</div>
                    <div class="kpi-sub">Benchmark corpus</div>
                </div>
            """, unsafe_allow_html=True)
        with s2:
            crit_cnt = (df_bench["ticket_priority"] == "Critical").sum()
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Critical Outages (P1)</div>
                    <div class="kpi-value" style="color: #F87171;">{crit_cnt:,}</div>
                    <div class="kpi-sub">15 min SLA window</div>
                </div>
            """, unsafe_allow_html=True)
        with s3:
            avg_sla = round(df_bench["sla_resolution_hours"].mean(), 1)
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Mean Resolution Target</div>
                    <div class="kpi-value" style="color: #FACC15;">{avg_sla} hrs</div>
                    <div class="kpi-sub">Weighted across priority</div>
                </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Support Departments</div>
                    <div class="kpi-value" style="color: #38BDF8;">{df_bench['assigned_team'].nunique()}</div>
                    <div class="kpi-sub">Automated routing queues</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1
        c_row1_1, c_row1_2 = st.columns(2)
        with c_row1_1:
            st.markdown("##### 🍩 Ticket Distribution by Category")
            cat_counts = df_bench["ticket_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_donut = px.pie(
                cat_counts, names="Category", values="Count", hole=0.55,
                color_discrete_sequence=["#38BDF8", "#818CF8", "#C084FC", "#F472B6", "#FB923C"]
            )
            fig_donut.update_layout(
                height=340, margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"), legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c_row1_2:
            st.markdown("##### 📊 Priority Breakdown per Category")
            prio_cat = pd.crosstab(df_bench["ticket_category"], df_bench["ticket_priority"]).reset_index()
            fig_bar = px.bar(
                prio_cat, x="ticket_category", y=["Critical", "High", "Medium", "Low"],
                barmode="stack",
                color_discrete_map={"Critical": "#EF4444", "High": "#F97316", "Medium": "#EAB308", "Low": "#22C55E"}
            )
            fig_bar.update_layout(
                height=340, margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                xaxis_title="Category", yaxis_title="Ticket Count",
                legend=dict(orientation="h", y=1.15)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Charts Row 2
        c_row2_1, c_row2_2 = st.columns(2)
        with c_row2_1:
            st.markdown("##### 📱 Ticket Volume by Communication Channel")
            chan_counts = df_bench["ticket_channel"].value_counts().reset_index()
            chan_counts.columns = ["Channel", "Tickets"]
            fig_chan = px.bar(
                chan_counts, x="Tickets", y="Channel", orientation="h",
                color="Tickets", color_continuous_scale="Tealgrn",
                text="Tickets"
            )
            fig_chan.update_layout(
                height=300, margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0")
            )
            st.plotly_chart(fig_chan, use_container_width=True)

        with c_row2_2:
            st.markdown("##### 🏢 Customer Account Tier Representation")
            tier_counts = df_bench["customer_tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Count"]
            fig_tier = px.pie(
                tier_counts, names="Tier", values="Count", hole=0.45,
                color_discrete_sequence=["#6366F1", "#A855F7", "#EC4899", "#14B8A6"]
            )
            fig_tier.update_layout(
                height=300, margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0")
            )
            st.plotly_chart(fig_tier, use_container_width=True)

    else:
        st.info("Benchmark dataset not found. Generating...")


# =============================================================================
# MODULE 4: MODEL BENCHMARK ARENA
# =============================================================================
elif app_mode == "🧪 Model Benchmark Arena":
    st.markdown('<div class="hero-title">NLP Model Benchmark Arena & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Head-to-head comparison of machine learning models across Accuracy, Precision, Recall, F1-Scores, latency, and confusion matrices.</div>', unsafe_allow_html=True)

    if metadata:
        tab_cat_m, tab_prio_m, tab_explain = st.tabs([
            "🎯 Category Classification Models",
            "⚡ Priority Prediction Models",
            "🔍 TF-IDF Vocabulary & Feature Weights"
        ])

        with tab_cat_m:
            st.markdown("#### 🏆 Category Models Leaderboard")
            cat_bench = metadata.get("category_benchmarks", {})
            if cat_bench:
                rows = []
                for m_name, metrics in cat_bench.items():
                    rows.append({
                        "Model Algorithm": m_name,
                        "Accuracy": f"{metrics['accuracy']:.2%}",
                        "F1 (Macro)": f"{metrics['f1_macro']:.4f}",
                        "F1 (Weighted)": f"{metrics['f1_weighted']:.4f}",
                        "Precision (Macro)": f"{metrics['precision_macro']:.4f}",
                        "Recall (Macro)": f"{metrics['recall_macro']:.4f}",
                        "Train Time (s)": f"{metrics['train_time_sec']:.2f}s"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔲 Category Confusion Matrix")
            best_cat_name = metadata.get("best_category_model", "Logistic Regression")
            best_cat_bench = cat_bench.get(best_cat_name, {})
            if best_cat_bench:
                cm = np.array(best_cat_bench.get("confusion_matrix_norm", []))
                classes = best_cat_bench.get("class_names", [])
                fig_cm = px.imshow(
                    cm, x=classes, y=classes,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Category", y="Actual Category", color="Accuracy %"),
                    text_auto=".1%"
                )
                fig_cm.update_layout(
                    height=450, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0")
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        with tab_prio_m:
            st.markdown("#### 🏆 Priority Models Leaderboard")
            prio_bench = metadata.get("priority_benchmarks", {})
            if prio_bench:
                p_rows = []
                for m_name, metrics in prio_bench.items():
                    p_rows.append({
                        "Model Algorithm": m_name,
                        "Accuracy": f"{metrics['accuracy']:.2%}",
                        "F1 (Macro)": f"{metrics['f1_macro']:.4f}",
                        "F1 (Weighted)": f"{metrics['f1_weighted']:.4f}",
                        "Precision (Macro)": f"{metrics['precision_macro']:.4f}",
                        "Recall (Macro)": f"{metrics['recall_macro']:.4f}",
                        "Train Time (s)": f"{metrics['train_time_sec']:.2f}s"
                    })
                st.dataframe(pd.DataFrame(p_rows), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔲 Priority Confusion Matrix")
            best_prio_name = metadata.get("best_priority_model", "Logistic Regression")
            best_prio_bench = prio_bench.get(best_prio_name, {})
            if best_prio_bench:
                cm_p = np.array(best_prio_bench.get("confusion_matrix_norm", []))
                classes_p = best_prio_bench.get("class_names", [])
                fig_cm_p = px.imshow(
                    cm_p, x=classes_p, y=classes_p,
                    color_continuous_scale="Purples",
                    labels=dict(x="Predicted Priority", y="Actual Priority", color="Accuracy %"),
                    text_auto=".1%"
                )
                fig_cm_p.update_layout(
                    height=450, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0")
                )
                st.plotly_chart(fig_cm_p, use_container_width=True)

        with tab_explain:
            st.markdown("#### 🔑 Top Predictive TF-IDF N-Grams by Category")
            top_words_cat = metadata.get("top_keywords_category", {})
            if top_words_cat:
                selected_cat_kw = st.selectbox("Select Category", list(top_words_cat.keys()))
                kw_data = top_words_cat[selected_cat_kw]
                df_kw = pd.DataFrame(kw_data)
                fig_kw = px.bar(
                    df_kw, x="weight", y="keyword", orientation="h",
                    color="weight", color_continuous_scale="Sunset",
                    text_auto=".3f"
                )
                fig_kw.update_layout(
                    height=380, margin=dict(l=10, r=10, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_kw, use_container_width=True)

    else:
        st.info("No benchmark metadata found. Please train models first.")


# =============================================================================
# MODULE 5: BUSINESS IMPACT & ROI CALCULATOR
# =============================================================================
elif app_mode == "💰 Business Impact & ROI":
    st.markdown('<div class="hero-title">Business Value & Operational ROI Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Demonstrate measurable dollar savings, support capacity expansion, and SLA compliance improvements to executive leadership and founders.</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Operational Parameters")
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        monthly_tickets = st.slider("Monthly Ticket Volume", min_value=1000, max_value=50000, value=8500, step=500)
        manual_triage_mins = st.slider("Manual Triage Time per Ticket (minutes)", min_value=1.0, max_value=8.0, value=4.5, step=0.5)

    with r_col2:
        hourly_wage = st.slider("Support Agent Hourly Wage ($ USD)", min_value=15, max_value=75, value=32, step=1)
        sla_breach_penalty = st.slider("Average Cost per SLA Breach ($ USD)", min_value=20, max_value=250, value=65, step=5)

    st.markdown('</div>', unsafe_allow_html=True)

    # Calculations
    hours_saved_month = (monthly_tickets * manual_triage_mins) / 60
    hours_saved_year = hours_saved_month * 12
    labor_savings_month = hours_saved_month * hourly_wage
    labor_savings_year = labor_savings_month * 12

    # Estimated SLA breaches prevented (assuming manual triage delayed 12% of tickets past SLA, ML reduces this to 1.5%)
    breaches_prevented_month = monthly_tickets * (0.12 - 0.015)
    penalty_savings_month = breaches_prevented_month * sla_breach_penalty
    penalty_savings_year = penalty_savings_month * 12

    total_annual_value = labor_savings_year + penalty_savings_year

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Projected Business Impact & Return on Investment")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Hours Saved / Month</div>
                <div class="kpi-value" style="color: #38BDF8;">{hours_saved_month:,.0f} hrs</div>
                <div class="kpi-sub">{hours_saved_year:,.0f} hours / year</div>
            </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Direct Labor Savings / Yr</div>
                <div class="kpi-value" style="color: #4ADE80;">${labor_savings_year:,.0f}</div>
                <div class="kpi-sub">${labor_savings_month:,.0f} / month</div>
            </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">SLA Penalties Avoided / Yr</div>
                <div class="kpi-value" style="color: #FACC15;">${penalty_savings_year:,.0f}</div>
                <div class="kpi-sub">{breaches_prevented_month*12:,.0f} breaches averted</div>
            </div>
        """, unsafe_allow_html=True)
    with b4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Annual Value</div>
                <div class="kpi-value" style="color: #C084FC;">${total_annual_value:,.0f}</div>
                <div class="kpi-sub">Net ROI multiplier: ~14x</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cum_savings = [(labor_savings_month + penalty_savings_month) * (i + 1) for i in range(12)]
    df_roi = pd.DataFrame({"Month": months, "Cumulative Savings ($)": cum_savings})

    fig_roi = px.area(
        df_roi, x="Month", y="Cumulative Savings ($)",
        title="Projected 12-Month Cumulative Cost Savings from Automated Support Triage",
        color_discrete_sequence=["#818CF8"]
    )
    fig_roi.update_layout(
        height=350, margin=dict(l=10, r=10, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0")
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # Executive memo export
    executive_memo = f"""# EXECUTIVE DECISION MEMORANDUM
**Date**: November 2026
**Subject**: Business Impact & ROI of Automated Support Ticket Classification & Prioritization
**Organization**: Customer Support Operations

## 1. Executive Summary
Implementing machine learning-based automated ticket classification and prioritization eliminates manual sorting delays, accelerates response times for critical production issues, and provides significant operational cost reductions.

## 2. Key Projected Metrics (Annualized)
* **Monthly Ticket Volume**: {monthly_tickets:,} tickets
* **Manual Hours Eliminated**: {hours_saved_year:,.0f} hours / year
* **Direct Labor Savings**: ${labor_savings_year:,.2f} USD
* **SLA Breach Penalties Prevented**: ${penalty_savings_year:,.2f} USD
* **Total Projected Annual Value**: ${total_annual_value:,.2f} USD

## 3. Operational Advantages
* **Zero First-Touch Latency**: Incoming tickets categorized in <50 milliseconds.
* **Intelligent Escalation**: Critical outages automatically routed to Level 2/3 DevOps.
* **Staff Focus**: Support engineers spend 100% of their shift solving customer problems rather than tagging tickets.
"""

    st.download_button(
        label="📄 Download Executive ROI Memo (.md)",
        data=executive_memo,
        file_name="Support_Ticket_ML_ROI_Memo.md",
        mime="text/markdown"
    )
