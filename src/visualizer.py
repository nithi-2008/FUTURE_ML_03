"""
Interactive Visualizer Module using Plotly.
Produces radar charts, ranking leaderboards, skill gap matrices, and candidate comparisons.
"""

from typing import Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ScreeningVisualizer:
    """
    Renders responsive Plotly visualizations for candidate screening,
    skill coverage radar charts, and comparative analytics.
    """

    @staticmethod
    def plot_leaderboard(ranked_candidates: List[Dict], top_n: int = 15) -> go.Figure:
        """
        Creates an interactive horizontal bar chart leaderboard of top candidates.
        """
        if not ranked_candidates:
            fig = go.Figure()
            fig.update_layout(title="No candidates to display")
            return fig

        top_candidates = ranked_candidates[:top_n]
        names = [f"#{c['rank']} {c['candidate_name']}" for c in top_candidates]
        scores = [c["composite_score"] for c in top_candidates]
        skill_scores = [c["skill_score"] for c in top_candidates]
        semantic_scores = [c["semantic_score"] for c in top_candidates]
        tiers = [c["tier"] for c in top_candidates]

        # Color mapping based on tier
        colors = []
        for tier in tiers:
            if "Highly" in tier:
                colors.append("#10B981")  # Emerald green
            elif "Potential" in tier:
                colors.append("#F59E0B")  # Amber
            else:
                colors.append("#EF4444")  # Rose red

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=names,
                orientation="h",
                marker=dict(color=colors, line=dict(color="#1F2937", width=1)),
                text=[f"{s}%" for s in scores],
                textposition="auto",
                hoverinfo="text",
                hovertext=[
                    f"<b>{c['candidate_name']}</b><br>"
                    f"Composite Fit: <b>{c['composite_score']}%</b><br>"
                    f"Skill Match: {c['skill_score']}%<br>"
                    f"Semantic Sim: {c['semantic_score']}%<br>"
                    f"Experience: {c['experience_years']} yrs<br>"
                    f"Status: {c['tier']}"
                    for c in top_candidates
                ]
            )
        )

        fig.update_layout(
            title=f"<b>Candidate Ranking Leaderboard (Top {len(top_candidates)})</b>",
            xaxis=dict(title="Composite Fit Score (%)", range=[0, 105], gridcolor="#E5E7EB"),
            yaxis=dict(autorange="reversed"),
            height=max(400, len(top_candidates) * 35),
            margin=dict(l=20, r=20, t=50, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )

        return fig

    @staticmethod
    def plot_radar_skills(candidate: Dict, jd_categories: Dict[str, int]) -> go.Figure:
        """
        Plots a radar chart showing candidate skill counts per domain vs required domain coverage.
        """
        cand_categories = candidate.get("skills_by_category", {})

        all_cats = sorted(list(set(list(jd_categories.keys()) + list(cand_categories.keys()))))
        if not all_cats:
            all_cats = ["Programming Languages", "AI & Machine Learning", "Cloud & DevOps", "Databases"]

        cand_values = [len(cand_categories.get(cat, [])) for cat in all_cats]
        jd_values = [jd_categories.get(cat, 1) for cat in all_cats]

        # Close loop for radar
        cats_closed = all_cats + [all_cats[0]]
        cand_closed = cand_values + [cand_values[0]]
        jd_closed = jd_values + [jd_values[0]]

        fig = go.Figure()

        # Job Requirement Baseline
        fig.add_trace(go.Scatterpolar(
            r=jd_closed,
            theta=cats_closed,
            fill="toself",
            name="Target Role Demand",
            line=dict(color="#6366F1", dash="dash"),
            fillcolor="rgba(99, 102, 241, 0.15)"
        ))

        # Candidate Coverage
        fig.add_trace(go.Scatterpolar(
            r=cand_closed,
            theta=cats_closed,
            fill="toself",
            name=f"{candidate.get('candidate_name', 'Candidate')} Skills",
            line=dict(color="#10B981", width=2),
            fillcolor="rgba(16, 185, 129, 0.25)"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(max(cand_values + jd_values, default=3) + 1, 4)])
            ),
            showlegend=True,
            title="<b>Domain Skill Competency Radar</b>",
            margin=dict(l=40, r=40, t=40, b=40),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        return fig

    @staticmethod
    def plot_skill_gap_summary(candidate: Dict) -> go.Figure:
        """
        Creates a breakdown donut chart of Matched vs Missing skills.
        """
        matched_req = len(candidate.get("matched_required", []))
        missing_req = len(candidate.get("missing_required", []))
        matched_pref = len(candidate.get("matched_preferred", []))
        missing_pref = len(candidate.get("missing_preferred", []))

        labels = ["Matched Required", "Missing Required", "Matched Preferred", "Missing Preferred"]
        values = [matched_req, missing_req, matched_pref, missing_pref]
        colors = ["#10B981", "#EF4444", "#3B82F6", "#F59E0B"]

        # Filter out 0 values
        non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
        if not non_zero:
            labels_f, values_f, colors_f = ["No Requirements"], [1], ["#9CA3AF"]
        else:
            labels_f = [x[0] for x in non_zero]
            values_f = [x[1] for x in non_zero]
            colors_f = [x[2] for x in non_zero]

        fig = go.Figure(
            go.Pie(
                labels=labels_f,
                values=values_f,
                hole=0.55,
                marker=dict(colors=colors_f, line=dict(color="#1F2937", width=1)),
                textinfo="label+value"
            )
        )

        fig.update_layout(
            title="<b>Skill Match & Gap Breakdown</b>",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        return fig

    @staticmethod
    def plot_multi_comparison(candidates: List[Dict]) -> go.Figure:
        """
        Side-by-side grouped bar chart comparing 2-5 candidates across all dimensions.
        """
        if not candidates:
            return go.Figure()

        names = [c["candidate_name"] for c in candidates]
        composite = [c["composite_score"] for c in candidates]
        skills = [c["skill_score"] for c in candidates]
        semantic = [c["semantic_score"] for c in candidates]
        exp = [c["experience_score"] for c in candidates]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Composite Fit",
            x=names,
            y=composite,
            marker_color="#6366F1",
            text=[f"{v}%" for v in composite],
            textposition="auto"
        ))

        fig.add_trace(go.Bar(
            name="Skill Match",
            x=names,
            y=skills,
            marker_color="#10B981",
            text=[f"{v}%" for v in skills],
            textposition="auto"
        ))

        fig.add_trace(go.Bar(
            name="Semantic Similarity",
            x=names,
            y=semantic,
            marker_color="#3B82F6",
            text=[f"{v}%" for v in semantic],
            textposition="auto"
        ))

        fig.add_trace(go.Bar(
            name="Experience Fit",
            x=names,
            y=exp,
            marker_color="#F59E0B",
            text=[f"{v}%" for v in exp],
            textposition="auto"
        ))

        fig.update_layout(
            barmode="group",
            title="<b>Side-by-Side Candidate Score Comparison</b>",
            yaxis=dict(title="Score (%)", range=[0, 110], gridcolor="#E5E7EB"),
            height=420,
            margin=dict(l=20, r=20, t=50, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig


visualizer = ScreeningVisualizer()
