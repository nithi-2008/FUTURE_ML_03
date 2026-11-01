"""
Production Inference & Decision-Support Engine
Handles single and batch ticket classification, priority prediction,
explainability keyword extraction, and SLA routing.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.text_preprocessor import clean_text, extract_ticket_signals, URGENCY_KEYWORDS


class TicketClassifier:
    """
    Unified Inference Engine for Ticket Category & Priority Triage.
    """
    _instance = None

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.cat_model = None
        self.prio_model = None
        self.vectorizer = None
        self.metadata = None
        self.load_models()

    def load_models(self):
        """Load serialized models, vectorizers, and evaluation metadata."""
        cat_path = os.path.join(self.models_dir, "category_model.joblib")
        prio_path = os.path.join(self.models_dir, "priority_model.joblib")
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        meta_path = os.path.join(self.models_dir, "model_metadata.json")

        if not (os.path.exists(cat_path) and os.path.exists(prio_path) and os.path.exists(vec_path)):
            raise FileNotFoundError(
                f"Model files not found in {self.models_dir}. Please run 'src/model_trainer.py' first."
            )

        self.cat_model = joblib.load(cat_path)
        self.prio_model = joblib.load(prio_path)
        self.vectorizer = joblib.load(vec_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def get_influential_keywords(self, cleaned_text: str, predicted_category: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identify which words in the user's ticket influenced the prediction most."""
        if not cleaned_text:
            return []

        tokens = list(set(cleaned_text.split()))
        vocab = self.vectorizer.vocabulary_
        matched_tokens = [t for t in tokens if t in vocab]

        if not matched_tokens:
            return []

        # Find category index
        cat_classes = list(self.cat_model.classes_)
        if predicted_category not in cat_classes:
            return []
        cat_idx = cat_classes.index(predicted_category)

        # Get coefficients
        coef = getattr(self.cat_model, "coef_", None)
        if coef is None and hasattr(self.cat_model, "calibrated_classifiers_"):
            coef = np.mean([clf.estimator.coef_ for clf in self.cat_model.calibrated_classifiers_], axis=0)

        token_scores = []
        for t in matched_tokens:
            feature_idx = vocab[t]
            if coef is not None and cat_idx < len(coef):
                score = float(coef[cat_idx][feature_idx])
            else:
                score = 1.0
            token_scores.append({"keyword": t, "importance": round(score, 4)})

        # Sort by importance descending
        token_scores.sort(key=lambda x: x["importance"], reverse=True)
        return token_scores[:top_n]

    def generate_draft_response(self, customer_name: str, category: str, priority: str) -> str:
        """Generate an AI draft response for the support agent to send."""
        first_name = customer_name.split()[0] if customer_name else "valued customer"

        templates = {
            "Technical Issue": (
                f"Hi {first_name},\n\n"
                f"Thank you for contacting our Technical Support team. We have received your report "
                f"and our engineering on-call team has been dispatched to investigate this issue.\n\n"
                f"Our telemetry systems are actively analyzing server logs. We will provide our next update "
                f"within our committed SLA window. If you have any stack traces or screenshots, please reply directly.\n\n"
                f"Best regards,\nTechnical Operations Team"
            ),
            "Billing & Invoicing": (
                f"Hi {first_name},\n\n"
                f"Thank you for reaching out regarding your account billing. We have escalated this ticket "
                f"to our Billing & Accounts Receivable team for immediate ledger review.\n\n"
                f"If an unauthorized or duplicate charge occurred, rest assured we will initiate a prompt refund "
                f"or invoice adjustment. We appreciate your patience while we review your transaction records.\n\n"
                f"Sincerely,\nBilling Operations Specialist"
            ),
            "Account & Security": (
                f"Hi {first_name},\n\n"
                f"Your security is our top priority. We have flagged your account request with our "
                f"Identity & Access Management desk.\n\n"
                f"For your protection, an administrator will verify your credentials and assist you with "
                f"restoring access or reconfiguring your authentication settings.\n\n"
                f"Stay safe,\nSecurity Operations Center"
            ),
            "Product Inquiry": (
                f"Hello {first_name},\n\n"
                f"Thanks for your interest in our platform! We love hearing questions and feature feedback from our community.\n\n"
                f"Our Solutions Architecture team has received your inquiry and is compiling the documentation and guidance you requested.\n\n"
                f"Warm regards,\nCustomer Success Team"
            ),
            "Cancellation & Refund": (
                f"Dear {first_name},\n\n"
                f"Thank you for getting in touch with our Account Services team. We have received your request "
                f"regarding your subscription and are reviewing your account status.\n\n"
                f"A dedicated retention specialist will follow up shortly to ensure your request is handled in full accordance "
                f"with our satisfaction guarantee policy.\n\n"
                f"Best,\nCustomer Care Desk"
            )
        }
        return templates.get(category, f"Hello {first_name},\n\nThank you for reaching out. A support specialist is reviewing your request.")

    def predict_ticket(
        self,
        subject: str,
        description: str,
        customer_name: str = "Customer",
        customer_tier: str = "Business",
        ticket_channel: str = "Web Portal"
    ) -> Dict[str, Any]:
        """
        End-to-end prediction for a single ticket with SLA assignment,
        signals, keyword explainability, and auto-response.
        """
        raw_combined = f"{subject} {description}".strip()
        cleaned = clean_text(raw_combined)
        signals = extract_ticket_signals(raw_combined)

        if not cleaned:
            # Fallback for empty input
            return {
                "category": "Product Inquiry",
                "category_confidence": 0.5,
                "category_probabilities": {},
                "priority": "Low",
                "priority_confidence": 0.5,
                "priority_probabilities": {},
                "escalated": False,
                "escalation_reason": "Default fallback",
                "sla_response_time": "12 hours",
                "sla_resolution_hours": 48,
                "assigned_team": "General Customer Care",
                "influential_keywords": [],
                "signals": signals,
                "draft_response": "Thank you for contacting us."
            }

        vec_input = self.vectorizer.transform([cleaned])

        # Category prediction
        cat_pred = str(self.cat_model.predict(vec_input)[0])
        if hasattr(self.cat_model, "predict_proba"):
            cat_probs = self.cat_model.predict_proba(vec_input)[0]
            cat_classes = list(self.cat_model.classes_)
            cat_prob_dict = {cat_classes[i]: round(float(cat_probs[i]), 4) for i in range(len(cat_classes))}
            cat_conf = round(float(max(cat_probs)), 4)
        else:
            cat_prob_dict = {cat_pred: 1.0}
            cat_conf = 1.0

        # Priority prediction
        prio_pred = str(self.prio_model.predict(vec_input)[0])
        if hasattr(self.prio_model, "predict_proba"):
            prio_probs = self.prio_model.predict_proba(vec_input)[0]
            prio_classes = list(self.prio_model.classes_)
            prio_prob_dict = {prio_classes[i]: round(float(prio_probs[i]), 4) for i in range(len(prio_classes))}
            prio_conf = round(float(max(prio_probs)), 4)
        else:
            prio_prob_dict = {prio_pred: 1.0}
            prio_conf = 1.0

        # Hybrid Decision Engine: Escalation Rules
        escalated = False
        escalation_reason = "Model predicted priority standard SLA."

        # Rule 1: High Urgency Keywords detected
        if signals["urgency_score"] >= 2 and prio_pred in ["Low", "Medium"]:
            prio_pred = "High"
            escalated = True
            escalation_reason = f"Automated escalation: Critical urgency terms found ({', '.join(signals['urgency_keywords_found'])})."

        # Rule 2: Enterprise Tier SLA upgrade
        if customer_tier == "Enterprise":
            if prio_pred == "Medium":
                prio_pred = "High"
                escalated = True
                escalation_reason = "Automated escalation: Enterprise Tier SLA tier policy."
            elif prio_pred == "Low":
                prio_pred = "Medium"
                escalated = True
                escalation_reason = "Automated escalation: Enterprise Tier SLA tier policy."

        # Rule 3: Extreme Panic / Outage phrase match
        lower_raw = raw_combined.lower()
        if any(term in lower_raw for term in ["production outage", "data breach", "hacked", "fraudulent", "system is down"]):
            if prio_pred != "Critical":
                prio_pred = "Critical"
                escalated = True
                escalation_reason = "Automated escalation: Critical production incident or security hazard detected."

        # SLA specifications
        sla_specs = {
            "Critical": {"response": "15 minutes", "resolution": 1, "color": "#FF4B4B"},
            "High": {"response": "1 hour", "resolution": 4, "color": "#FFA500"},
            "Medium": {"response": "4 hours", "resolution": 24, "color": "#F0C000"},
            "Low": {"response": "12 hours", "resolution": 48, "color": "#28A745"}
        }
        sla_info = sla_specs.get(prio_pred, sla_specs["Medium"])

        # Routing department
        dept_map = {
            "Technical Issue": "Tier 2 Technical Support & DevOps",
            "Billing & Invoicing": "Billing & Finance Operations",
            "Account & Security": "Identity & Security Operations",
            "Product Inquiry": "Customer Success & Solutions Architecture",
            "Cancellation & Refund": "Retention & Account Management"
        }
        assigned_team = dept_map.get(cat_pred, "General Customer Care")

        # Explainability keywords
        keywords = self.get_influential_keywords(cleaned, cat_pred, top_n=6)

        # Draft response
        draft = self.generate_draft_response(customer_name, cat_pred, prio_pred)

        return {
            "category": cat_pred,
            "category_confidence": cat_conf,
            "category_probabilities": cat_prob_dict,
            "priority": prio_pred,
            "priority_confidence": prio_conf,
            "priority_probabilities": prio_prob_dict,
            "priority_color": sla_info["color"],
            "escalated": escalated,
            "escalation_reason": escalation_reason,
            "sla_response_time": sla_info["response"],
            "sla_resolution_hours": sla_info["resolution"],
            "assigned_team": assigned_team,
            "influential_keywords": keywords,
            "signals": signals,
            "draft_response": draft
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        High-throughput batch inference over a DataFrame of tickets.
        Expects columns: 'ticket_subject' and 'ticket_description' (or similar).
        """
        df_out = df.copy()

        # Identify subject and description columns
        subj_col = None
        desc_col = None
        for col in df_out.columns:
            lower = col.lower()
            if "subject" in lower or "title" in lower:
                subj_col = col
            elif "desc" in lower or "body" in lower or "text" in lower:
                desc_col = col

        if subj_col is None and desc_col is None:
            # Fallback to first text column
            first_text_col = df_out.select_dtypes(include="object").columns[0]
            raw_texts = df_out[first_text_col].fillna("").astype(str)
        elif subj_col and desc_col:
            raw_texts = df_out[subj_col].fillna("").astype(str) + " " + df_out[desc_col].fillna("").astype(str)
        else:
            col = subj_col or desc_col
            raw_texts = df_out[col].fillna("").astype(str)

        cleaned_texts = raw_texts.apply(clean_text)
        vec_inputs = self.vectorizer.transform(cleaned_texts)

        cat_preds = self.cat_model.predict(vec_inputs)
        prio_preds = self.prio_model.predict(vec_inputs)

        cat_confs = []
        if hasattr(self.cat_model, "predict_proba"):
            cat_probs = self.cat_model.predict_proba(vec_inputs)
            cat_confs = np.max(cat_probs, axis=1).round(4)
        else:
            cat_confs = [1.0] * len(df_out)

        prio_confs = []
        if hasattr(self.prio_model, "predict_proba"):
            prio_probs = self.prio_model.predict_proba(vec_inputs)
            prio_confs = np.max(prio_probs, axis=1).round(4)
        else:
            prio_confs = [1.0] * len(df_out)

        # Department & SLA maps
        dept_map = {
            "Technical Issue": "Tier 2 Technical Support & DevOps",
            "Billing & Invoicing": "Billing & Finance Operations",
            "Account & Security": "Identity & Security Operations",
            "Product Inquiry": "Customer Success & Solutions Architecture",
            "Cancellation & Refund": "Retention & Account Management"
        }
        sla_hours_map = {"Critical": 1, "High": 4, "Medium": 24, "Low": 48}

        df_out["predicted_category"] = cat_preds
        df_out["category_confidence"] = cat_confs
        df_out["predicted_priority"] = prio_preds
        df_out["priority_confidence"] = prio_confs
        df_out["assigned_team"] = [dept_map.get(c, "General Customer Care") for c in cat_preds]
        df_out["sla_target_hours"] = [sla_hours_map.get(p, 24) for p in prio_preds]

        return df_out


# Singleton helper
def get_classifier(models_dir: str = "models") -> TicketClassifier:
    """Return a cached TicketClassifier instance."""
    if TicketClassifier._instance is None:
        TicketClassifier._instance = TicketClassifier(models_dir=models_dir)
    return TicketClassifier._instance
