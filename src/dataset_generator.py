"""
Enterprise Support Ticket Dataset Generator & Loader
Generates realistic, diverse multi-category support tickets with priorities and metadata.
"""

import os
import random
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Seed for reproducibility
RANDOM_SEED = 42

CATEGORIES = [
    "Technical Issue",
    "Billing & Invoicing",
    "Account & Security",
    "Product Inquiry",
    "Cancellation & Refund"
]

PRIORITIES = ["Critical", "High", "Medium", "Low"]

CHANNELS = ["Email", "Web Portal", "Live Chat", "Mobile App", "API / Webhook"]
CUSTOMER_TIERS = ["Enterprise", "Business", "Startup", "Free Tier"]

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Robin",
    "Casey", "Jamie", "Riley", "Cameron", "Dakota", "Jesse", "Avery", "Logan",
    "David", "Sarah", "Michael", "Elena", "Marcus", "Priya", "Chen", "Fatima"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"
]
DOMAINS = ["acme-corp.com", "fintech-scale.io", "cloudops.net", "innovate.org", "vertextech.com", "globalretail.com", "example.com"]

# Rich domain corpora for ticket generation
TEMPLATES: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "Technical Issue": {
        "Critical": [
            {
                "subject": "CRITICAL OUTAGE: Production API returning 500 Internal Server Errors",
                "body": "Our entire production cluster is down. API endpoints are responding with 500 status codes since 14:00 UTC. Over 50,000 of our active users cannot checkout. Please escalate to Level 3 DevOps immediately! This is a complete system block."
            },
            {
                "subject": "Emergency: Primary PostgreSQL Database Deadlock & Connection Pool Exhausted",
                "body": "We are observing catastrophic database lockouts. All worker nodes are failing with 'too many connections' and transactions are timing out. Our core payment backend is completely offline. Immediate response needed!"
            },
            {
                "subject": "Severe Data Corruption detected after migration to v4.2",
                "body": "After the automated patch to v4.2 last night, customer records are showing garbled foreign keys and NULL account balances. We had to halt live ingest to avoid further loss. Please roll back or patch immediately."
            },
            {
                "subject": "Production Webhooks completely stalled - queue backed up by 250k events",
                "body": "All outbound webhooks to our fulfillment partners stopped firing 45 minutes ago. Event queue is overflowing. Downstream operations have come to a standstill. Treat this as P1 critical outage."
            },
            {
                "subject": "Fatal crash loop in mobile SDK after iOS 18 update",
                "body": "Our mobile application is crashing on launch for 100% of iOS users with SIGABRT in the auth module. App Store reviews are tanking. We need an emergency hotfix build right now."
            }
        ],
        "High": [
            {
                "subject": "High latency on search index - queries taking over 12 seconds",
                "body": "Elasticsearch queries have degraded from 80ms to over 12 seconds since this morning. Search functionality is effectively unusable for our enterprise customers during peak hours."
            },
            {
                "subject": "Intermittent SSL Handshake Failures on custom domain integration",
                "body": "Several of our clients are reporting ERR_SSL_PROTOCOL_ERROR when navigating to our hosted portal. It seems the automated Let's Encrypt certificate renewal failed silently."
            },
            {
                "subject": "File upload endpoint failing for attachments over 5MB",
                "body": "Users are unable to upload standard PDF receipts. The gateway drops the socket after 30 seconds with 504 Gateway Timeout. This is blocking our accounting team from closing end-of-month files."
            },
            {
                "subject": "Background worker memory leak causing daily OOM kills",
                "body": "The report processing worker container crashes every 4 hours due to an out-of-memory error. Batch exports are delayed by several hours each day. Please provide a memory diagnostic."
            },
            {
                "subject": "Real-time WebSocket disconnections every 60 seconds",
                "body": "Our live collaboration canvas loses WebSocket connectivity every minute and forces a hard browser reload. Multiple enterprise clients have raised escalations."
            }
        ],
        "Medium": [
            {
                "subject": "Minor styling glitch on mobile browser navigation bar",
                "body": "When viewing the dashboard on Safari iOS in landscape mode, the dropdown menu overlaps with the header logo. It functions if clicked carefully, but looks unpolished."
            },
            {
                "subject": "CSV Export encoding problem with UTF-8 characters",
                "body": "Exporting customer reports with accented characters (e.g. é, ö, ñ) displays question marks instead in Excel. Works fine in Google Sheets, so seems like missing BOM header."
            },
            {
                "subject": "Webhook retry interval not matching documented exponential backoff",
                "body": "We noticed failed webhook retries are sent every 30 seconds instead of following the 1m, 5m, 15m schedule described in your developer documentation."
            },
            {
                "subject": "User profile avatar image not updating immediately after upload",
                "body": "When changing profile picture, the browser continues serving the cached thumbnail for several hours unless we force clear the local browser cache."
            }
        ],
        "Low": [
            {
                "subject": "Typo in API documentation for endpoint /v2/analytics/reports",
                "body": "In the documentation parameters table, 'timestamp_start' is spelled as 'timstamp_start'. Just flagging this minor documentation error for your tech writers."
            },
            {
                "subject": "Question regarding supported TLS cipher suites",
                "body": "Our security audit team is reviewing third-party vendors and asked whether your ingress load balancers support TLS 1.3 exclusively or allow 1.2 fallback."
            },
            {
                "subject": "Feature request: Keyboard shortcut for toggling sidebar",
                "body": "It would be great if we could press Ctrl+B or Cmd+B to collapse the sidebar when working on smaller laptop screens."
            }
        ]
    },
    "Billing & Invoicing": {
        "Critical": [
            {
                "subject": "URGENT BILLING: Unauthorized recurring charge of $12,500 on business card",
                "body": "We just noticed an unauthorized charge of $12,500 processed on our corporate Amex card today! We are on the $499 monthly plan. Reverse this fraudulent charge immediately or our bank will file a chargeback."
            },
            {
                "subject": "Account suspended due to billing system error despite successful wire transfer",
                "body": "Our entire company account has been locked with a 'Payment Past Due' banner, even though our $45,000 annual wire transfer cleared last Monday. Our team cannot work! Restore access immediately."
            },
            {
                "subject": "Payment processor looping: Customer charged 5 times for single upgrade",
                "body": "A checkout glitch triggered 5 sequential charges of $1,200 on our company account within 10 minutes. Please void the duplicate transactions and notify your billing team immediately."
            }
        ],
        "High": [
            {
                "subject": "Incorrect prorated calculation after upgrading seats",
                "body": "We added 5 new seats mid-cycle, but your invoice charged us for a full 12 months rather than the 18 remaining days of the current billing period. Please correct invoice #INV-88219."
            },
            {
                "subject": "Credit card auto-pay failing with error code 'Do Not Honor'",
                "body": "Our primary corporate Visa was rejected during renewal. Our bank confirms there are no restrictions on their end. We need an extension before the grace period ends so service isn't disrupted."
            },
            {
                "subject": "Tax Exempt status not applied to recent enterprise invoice",
                "body": "We submitted our 501(c)(3) non-profit tax exemption certificate two weeks ago, but our latest invoice still included $850 in state sales tax. Please re-issue invoice without tax."
            }
        ],
        "Medium": [
            {
                "subject": "Need updated VAT / Tax ID added to invoice receipts",
                "body": "Could you please add our European VAT number GB992817261 to all past invoices for Q3 and Q4? Our finance team requires it for quarterly tax filing."
            },
            {
                "subject": "Request for centralized billing email address change",
                "body": "Please change the primary billing notification contact from john@company.com to billing-ops@company.com so our automated AP inbox receives invoices."
            },
            {
                "subject": "Itemized breakdown request for monthly usage overage charges",
                "body": "Our latest bill had an extra $140 labeled 'storage overage'. Could you provide a daily breakdown of which projects exceeded the 500GB quota?"
            }
        ],
        "Low": [
            {
                "subject": "Inquiry regarding annual billing payment methods (ACH / Wire transfer)",
                "body": "We are considering switching from monthly credit card billing to an annual contract. Do you accept ACH direct debit or international wire transfer?"
            },
            {
                "subject": "Where can I download receipts for the past 12 months?",
                "body": "Hi, our accountants are conducting an annual audit. Can you point me to where I can download all invoices from 2025 in a single ZIP file?"
            }
        ]
    },
    "Account & Security": {
        "Critical": [
            {
                "subject": "SECURITY INCIDENT: Suspicious logins from unrecognized foreign IP addresses",
                "body": "Our admin dashboard shows 40 successful login attempts from an unknown Russian IP within 15 minutes. We suspect our API master secret key has been compromised. Invalidate all session tokens and revoke keys immediately!"
            },
            {
                "subject": "Complete Org Lockout: Primary admin 2FA device destroyed in accident",
                "body": "Our sole organization administrator lost their physical phone containing Google Authenticator. Backup codes were saved inside the account. All 120 employees are locked out. Urgent executive identity verification needed!"
            },
            {
                "subject": "Data Breach Alert: Unauthorized user granted Super-Admin permissions",
                "body": "An audit log alert triggered showing an external email address was assigned Super-Admin permissions without our knowledge. Please lock the organization workspace immediately to prevent data exfiltration."
            }
        ],
        "High": [
            {
                "subject": "SAML SSO integration failing with 'Invalid Signature' error",
                "body": "Our Okta single sign-on integration broke this morning after our IdP certificate renewal. 300+ employees cannot authenticate through company portal. Please review SAML metadata assertion."
            },
            {
                "subject": "Account locked after entering password incorrectly - reset email not arriving",
                "body": "My executive account was locked after three incorrect password entries. I clicked 'Forgot Password' three times over the past 2 hours, but no reset link arrives in my inbox or spam."
            },
            {
                "subject": "Former employee still able to access proprietary workspaces",
                "body": "We deactivated a departing contractor in Google Workspace, but their session tokens inside your platform appear to remain valid. We need an immediate forced token invalidation."
            }
        ],
        "Medium": [
            {
                "subject": "How to configure SCIM automated user provisioning with Azure AD?",
                "body": "We are setting up automated user onboarding via Azure Active Directory. Could you verify our SCIM bearer token configuration and endpoint URL?"
            },
            {
                "subject": "Transfer organization ownership to new team lead",
                "body": "Our previous account owner has transitioned to another department. Both the current owner and new owner are CC'd on this ticket to approve ownership transfer."
            },
            {
                "subject": "Enforce mandatory Two-Factor Authentication (2FA) for all workspace members",
                "body": "We want to enforce 2FA across all 45 users in our workspace. Does your admin console have a toggle to require 2FA before next login?"
            }
        ],
        "Low": [
            {
                "subject": "How do I change my registered email address?",
                "body": "Hello, my company recently rebranded and my email domain changed from oldname.com to newbrand.com. What are the steps to update my login email?"
            },
            {
                "subject": "Inquiry regarding session timeout configuration",
                "body": "Is it possible to set the inactivity session logout duration to 15 minutes instead of the default 24 hours to comply with internal policy?"
            }
        ]
    },
    "Product Inquiry": {
        "Critical": [
            {
                "subject": "Pre-Sale Blocker: Enterprise compliance audit requires SOC2 Type II report",
                "body": "We are finalizing a $180,000 annual contract with your sales director today, but our Chief Information Security Officer will not sign without reviewing your latest SOC2 Type II and ISO 27001 reports under NDA. Urgent."
            }
        ],
        "High": [
            {
                "subject": "Upcoming API deprecation timeline and breaking changes verification",
                "body": "We received an email that v1 REST API will sunset in 30 days. We need technical confirmation whether the payload schema for /orders has breaking structural changes."
            },
            {
                "subject": "Inquiry on maximum concurrent request limits for Enterprise tier",
                "body": "We are launching a nationwide marketing campaign next Friday and anticipate 5,000 requests per second. Does our Enterprise tier auto-scale or do we need dedicated provisioning?"
            }
        ],
        "Medium": [
            {
                "subject": "Does the platform support automated syncing with Snowflake or BigQuery?",
                "body": "Our analytics team is building a centralized data warehouse. Do you have a native connector for Snowflake or do we need to build an ETL pipeline via your REST API?"
            },
            {
                "subject": "Question regarding webhook payload signature verification (HMAC-SHA256)",
                "body": "Could you provide a Python or Node.js code example for verifying the X-Signature header sent with incoming webhook events?"
            },
            {
                "subject": "Differences between Business and Enterprise SLA tiers",
                "body": "Could someone from the solutions team walk us through the difference in response time guarantees between Business (99.5%) and Enterprise (99.99%)?"
            }
        ],
        "Low": [
            {
                "subject": "Feature suggestion: Dark mode theme for analytics dashboard",
                "body": "Hi there! Our night-shift monitoring engineers love the tool, but a dark mode interface would be much easier on their eyes. Is this on your product roadmap?"
            },
            {
                "subject": "Can custom fields be reordered in the ticket submission form?",
                "body": "We created 4 custom dropdown fields. Is there drag-and-drop capability to change the order in which they appear to our end users?"
            },
            {
                "subject": "General question about mobile app availability for Android",
                "body": "We see your iOS app in the Apple App Store. Is there a public beta or timeline for the Google Play Store Android release?"
            }
        ]
    },
    "Cancellation & Refund": {
        "Critical": [
            {
                "subject": "DEMAND IMMEDIATE REFUND: Charged $3,600 after explicit cancellation confirmation",
                "body": "I received written confirmation from your support rep on October 12th that our subscription was terminated. Today you processed an automatic renewal for $3,600! This is unlawful unauthorized billing. Issue a 100% full refund immediately."
            },
            {
                "subject": "Critical account cancellation and GDPR right-to-be-forgotten data wipe",
                "body": "We have legally terminated our agreement due to contractual breach. Under GDPR Article 17, we demand the immediate deletion of all customer data, backups, and user logs within 48 hours."
            }
        ],
        "High": [
            {
                "subject": "Refund request within 30-day money-back guarantee period",
                "body": "We signed up 18 days ago. Unfortunately, the software lacks key inventory integrations needed for our business. Please process our refund under your 30-day guarantee."
            },
            {
                "subject": "Downgrade plan from Enterprise to Starter effective immediately",
                "body": "Due to company downsizing, we need to downgrade from 100 seats to 10 seats before our upcoming billing renewal on the 1st of next month."
            },
            {
                "subject": "Subscription cancellation button disabled in dashboard",
                "body": "I am trying to cancel our monthly plan before tomorrow's renewal, but the 'Cancel Plan' button in billing settings is greyed out. Please cancel it manually from your end so I am not billed."
            }
        ],
        "Medium": [
            {
                "subject": "Inquiry on cancellation terms and data export grace period",
                "body": "If we decide to cancel our subscription at the end of the quarter, how long will our historic data remain accessible for CSV export before the workspace is deleted?"
            },
            {
                "subject": "Pause subscription for 3 months during seasonal offseason",
                "body": "We are a seasonal ski resort business and would like to pause our account from May through August rather than cancelling completely. Is account pausing supported?"
            }
        ],
        "Low": [
            {
                "subject": "Feedback regarding our reason for cancelling",
                "body": "Just submitting our exit survey feedback. Your platform was great, but our company was acquired by a parent firm that standardizes on Salesforce. Thank you for the good support."
            },
            {
                "subject": "Notice of non-renewal at the end of the annual contract",
                "body": "Please record this notice that we will not be renewing our annual contract when it expires in 90 days. Please do not auto-renew."
            }
        ]
    }
}

# Modifiers, technical nouns, and fillers to expand tickets into thousands of unique examples
ISSUE_VARIANTS = [
    "We have noticed this across both staging and production environments.",
    "This started happening right after yesterday's maintenance window.",
    "Our monitoring dashboards fired multiple alerts over the last 30 minutes.",
    "Multiple team members have confirmed reproducing this on Chrome and Firefox.",
    "We tried clearing cache, updating cookies, and testing in incognito mode with no success.",
    "This is impacting our core business operations during peak European trading hours.",
    "Please treat this with the highest priority as our executives are tracking this.",
    "Let us know if you need HAR logs, stack traces, or console recordings.",
    "We have already verified that our internal network and firewalls are operating normally.",
    "Could someone from the on-call engineering team provide an ETA on resolution?",
    "We are receiving complaints from our end-users and customers as well.",
    "Everything was working smoothly until the latest version update was pushed.",
    "Looking forward to your prompt response.",
    "Kindly update us as soon as an engineer investigates the root cause."
]


def generate_single_ticket(ticket_id: int, category: str, priority: str) -> Dict[str, Any]:
    """Create a single realistic ticket record with text and metadata."""
    category_templates = TEMPLATES.get(category, {})
    priority_templates = category_templates.get(priority, [])

    if not priority_templates:
        # Fallback to any priority if specific one not listed
        all_templates = [t for p_list in category_templates.values() for t in p_list]
        template = random.choice(all_templates)
    else:
        template = random.choice(priority_templates)

    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    company_domain = random.choice(DOMAINS)
    email = f"{first_name.lower()}.{last_name.lower()}@{company_domain}"
    channel = random.choice(CHANNELS)
    tier = random.choice(CUSTOMER_TIERS)

    # Enterprise customers have higher priority tendency
    if tier == "Enterprise" and priority in ["Medium", "Low"] and random.random() < 0.35:
        priority = "High"

    # Assemble subject and body with realistic modifiers
    subject = template["subject"]
    body = template["body"]

    # Add 1-2 random modifier sentences to create realistic text variability
    extra_modifiers = random.sample(ISSUE_VARIANTS, k=random.randint(1, 3))
    full_description = f"{body} {' '.join(extra_modifiers)}"

    # SLA resolution hours based on priority
    sla_map = {
        "Critical": 1,
        "High": 4,
        "Medium": 24,
        "Low": 48
    }
    sla_hours = sla_map.get(priority, 24)

    # Department routing
    dept_map = {
        "Technical Issue": "Tier 2 Technical Support & DevOps",
        "Billing & Invoicing": "Billing & Finance Operations",
        "Account & Security": "Identity & Security Operations",
        "Product Inquiry": "Customer Success & Solutions Architecture",
        "Cancellation & Refund": "Retention & Account Management"
    }
    assigned_team = dept_map.get(category, "General Customer Care")

    # Sentiment proxy
    if priority == "Critical":
        sentiment = "Highly Negative"
    elif priority == "High":
        sentiment = "Negative"
    elif priority == "Medium":
        sentiment = "Neutral"
    else:
        sentiment = "Positive / Inquisitive"

    return {
        "ticket_id": f"TCK-{ticket_id:05d}",
        "customer_name": f"{first_name} {last_name}",
        "customer_email": email,
        "customer_tier": tier,
        "ticket_channel": channel,
        "ticket_subject": subject,
        "ticket_description": full_description,
        "ticket_category": category,
        "ticket_priority": priority,
        "sla_resolution_hours": sla_hours,
        "assigned_team": assigned_team,
        "customer_sentiment": sentiment
    }


def generate_support_dataset(total_tickets: int = 5000, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a balanced, high-quality support tickets dataset."""
    random.seed(random_state)
    np.random.seed(random_state)

    tickets = []
    tickets_per_category = total_tickets // len(CATEGORIES)

    ticket_id = 1
    for category in CATEGORIES:
        for _ in range(tickets_per_category):
            # Realistic priority distribution: Critical (15%), High (30%), Medium (35%), Low (20%)
            priority = np.random.choice(
                PRIORITIES,
                p=[0.15, 0.30, 0.35, 0.20]
            )
            ticket_data = generate_single_ticket(ticket_id, category, priority)
            tickets.append(ticket_data)
            ticket_id += 1

    df = pd.DataFrame(tickets)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def ensure_data_directories(base_dir: str = "."):
    """Ensure data/ and models/ directories exist."""
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)


def build_and_save_datasets(data_dir: str = "data"):
    """Generate and persist the benchmark dataset and sample batch dataset."""
    os.makedirs(data_dir, exist_ok=True)
    main_path = os.path.join(data_dir, "customer_support_tickets.csv")
    sample_path = os.path.join(data_dir, "sample_tickets_batch.csv")

    print(f"Generating benchmark customer support tickets dataset (5,000 tickets)...")
    df_main = generate_support_dataset(total_tickets=5000, random_state=42)
    df_main.to_csv(main_path, index=False)
    print(f"[OK] Saved benchmark dataset to: {main_path} (Shape: {df_main.shape})")

    # Generate 50 realistic unlabelled sample tickets for the UI batch testing demo
    print("Generating sample unlabelled tickets for UI batch ingestion demo...")
    df_sample = generate_support_dataset(total_tickets=50, random_state=99)
    # Provide the raw input columns that a user would upload in a CSV
    ui_batch_df = df_sample[["ticket_id", "customer_name", "customer_email", "customer_tier", "ticket_channel", "ticket_subject", "ticket_description"]].copy()
    ui_batch_df.to_csv(sample_path, index=False)
    print(f"[OK] Saved UI sample batch to: {sample_path} (Shape: {ui_batch_df.shape})")

    return df_main, ui_batch_df


if __name__ == "__main__":
    build_and_save_datasets()
