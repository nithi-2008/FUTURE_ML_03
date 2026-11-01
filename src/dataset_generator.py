"""
Dataset and Sample Resume Generator.
Generates realistic resumes across diverse engineering and data domains,
exports a CSV dataset, produces PDF/TXT sample resumes, and fits model artifacts.
"""

import json
import os
import random
from typing import Dict, List
import joblib
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sklearn.feature_extraction.text import TfidfVectorizer


# Domain profiles and templates
DOMAINS = {
    "Machine Learning Engineer": {
        "titles": ["Senior Machine Learning Engineer", "ML Research Engineer", "AI Systems Engineer", "Applied ML Scientist"],
        "core_skills": ["Python", "PyTorch", "TensorFlow", "Scikit-Learn", "FastAPI", "Docker", "AWS", "Git"],
        "extra_skills": ["Kubernetes", "MLflow", "Deep Learning", "NLP", "LLMs", "CI/CD", "PostgreSQL", "Pandas", "NumPy", "Hugging Face"],
        "degrees": ["Master's Degree", "Doctorate (Ph.D.)", "Bachelor's Degree"],
        "exp_range": (3, 8),
        "templates": [
            "Experienced Machine Learning Engineer specializing in designing and deploying scalable deep learning pipelines. Developed high-throughput NLP classification microservices using PyTorch, Hugging Face, and FastAPI, containerized with Docker and orchestrated on AWS ECS. Implemented distributed model tracking using MLflow and automated CI/CD retraining workflows with GitHub Actions. Deep background in Scikit-Learn feature engineering, Pandas data transformations, and Git version control.",
            "Applied AI Engineer with extensive background in building computer vision and large language model architectures. Built real-time inference services with Python, PyTorch, and Docker on AWS infrastructure. Managed large-scale model experimentation with Weights & Biases and Scikit-Learn. Passionate about MLOps, Kubernetes deployment, and high-performance REST APIs.",
            "Machine Learning practitioner focused on predictive algorithms, customer churn estimation, and deep learning architectures. Fluent in Python, TensorFlow, PyTorch, Scikit-Learn, and SQL. Spearheaded cloud model deployment using Docker, FastAPI, and PostgreSQL on AWS. Collaborated with cross-functional engineering teams using Agile methodologies."
        ]
    },
    "Full Stack Developer": {
        "titles": ["Senior Full Stack Developer", "Full Stack Engineer", "Web Applications Engineer", "React & Node.js Developer"],
        "core_skills": ["JavaScript", "TypeScript", "React", "Node.js", "Express.js", "PostgreSQL", "REST APIs", "Git"],
        "extra_skills": ["Next.js", "MongoDB", "Docker", "AWS", "GraphQL", "Tailwind CSS", "Agile", "HTML5", "CSS3", "Unit Testing"],
        "degrees": ["Bachelor's Degree", "Master's Degree"],
        "exp_range": (2, 7),
        "templates": [
            "Dynamic Full Stack Engineer with extensive experience developing full-cycle web applications. Proficient in crafting responsive frontend user interfaces using React, TypeScript, Next.js, and Tailwind CSS. Built high-concurrency backend services using Node.js, Express.js, and GraphQL, integrating PostgreSQL and MongoDB datastores. Automated end-to-end unit testing with Jest and deployed cloud workloads with Docker and AWS.",
            "Web Applications Developer skilled in modern JavaScript, TypeScript, React, and Node.js. Architected scalable REST APIs with Express.js and integrated secure OAuth authentication with PostgreSQL databases. Leveraged Git for version control, collaborated in Agile sprints, and managed cloud deployments with Docker on AWS.",
            "Versatile Software Engineer with strong front-end and back-end proficiencies. Designed interactive client portals using React, Redux, and HTML5/CSS3. Engineered backend microservices using Node.js, Express.js, and REST APIs backed by MongoDB and Redis caching. Adept at CI/CD workflows with GitHub Actions."
        ]
    },
    "Cloud & DevOps Engineer": {
        "titles": ["Lead Cloud DevOps Engineer", "Site Reliability Engineer (SRE)", "Infrastructure Automation Engineer", "Platform Systems Architect"],
        "core_skills": ["AWS", "Docker", "Kubernetes", "Linux", "Terraform", "CI/CD", "GitHub Actions", "Bash"],
        "extra_skills": ["Ansible", "Prometheus", "Grafana", "Python", "Nginx", "Azure", "System Design", "Git"],
        "degrees": ["Bachelor's Degree", "Associate / Diploma", "Master's Degree"],
        "exp_range": (3, 9),
        "templates": [
            "DevOps and Cloud Infrastructure Engineer with a proven track record of maintaining 99.99% uptime for multi-tenant SaaS environments. Master of Infrastructure as Code using Terraform and Ansible across AWS and Azure cloud platforms. Configured multi-region Kubernetes (EKS) clusters, automated enterprise CI/CD pipelines with GitHub Actions and Jenkins, and built centralized monitoring dashboards with Prometheus and Grafana. Advanced Linux administration and Bash automation scripting.",
            "Site Reliability Engineer dedicated to zero-downtime releases, scalable container orchestration, and security hardening. Managed containerized clusters using Docker, Kubernetes, and Helm on AWS. Created automated build and release workflows with CI/CD and GitHub Actions. Proficient in Python scripting for infrastructure tooling, Nginx load balancing, and Linux troubleshooting.",
            "Cloud Systems Architect specializing in AWS enterprise cloud migrations and GitOps automation. Authored declarative Terraform modules for VPC, EKS, RDS, and CloudFront. Automated container image vulnerability scanning and deployments using Docker and GitHub Actions. Monitored cluster telemetry with Prometheus and Grafana."
        ]
    },
    "Data Scientist": {
        "titles": ["Senior Data Scientist", "Analytics Research Scientist", "Predictive Analytics Specialist", "Quantitative Data Scientist"],
        "core_skills": ["Python", "SQL", "Pandas", "NumPy", "Scikit-Learn", "NLP", "Statistical Analysis", "Git"],
        "extra_skills": ["spaCy", "NLTK", "Tableau", "A/B Testing", "Machine Learning", "BigQuery", "Data Analysis", "Matplotlib", "Seaborn"],
        "degrees": ["Master's Degree", "Doctorate (Ph.D.)", "Bachelor's Degree"],
        "exp_range": (2, 7),
        "templates": [
            "Data Scientist passionate about mining insights from complex unstructured and tabular business data. Highly skilled in Python statistical computing using Pandas, NumPy, and Scikit-Learn. Engineered text mining and natural language processing pipelines using spaCy and NLTK for entity recognition and topic modeling. Designed rigorous A/B testing frameworks, evaluated hypothesis tests, and visualized executive reports using Tableau and Seaborn. Advanced SQL optimization on Snowflake and BigQuery.",
            "Quantitative Scientist with deep expertise in predictive modeling, regression, clustering, and customer lifetime value prediction. Built robust statistical models with Scikit-Learn and Python. Extracted and normalized billions of records using SQL and Pandas. Translated complex mathematical findings into intuitive charts using Matplotlib and Tableau.",
            "Data Science specialist focusing on conversational NLP and predictive analytics. Developed sentiment classification engines utilizing NLTK, spaCy, and Scikit-Learn. Designed A/B tests to measure feature adoption and conducted exploratory data analysis on multi-terabyte datasets using BigQuery and SQL."
        ]
    },
    "Data Analyst": {
        "titles": ["Lead Business Intelligence Analyst", "Senior Data Analyst", "Operations Analytics Specialist", "Financial Data Analyst"],
        "core_skills": ["SQL", "Power BI", "Tableau", "Excel", "Data Analysis", "Communication", "Problem Solving"],
        "extra_skills": ["Python", "Pandas", "PostgreSQL", "Snowflake", "A/B Testing", "Statistical Analysis", "Git"],
        "degrees": ["Bachelor's Degree", "Master's Degree"],
        "exp_range": (1, 6),
        "templates": [
            "Data and Business Intelligence Analyst experienced in translating raw transactional data into executive decision dashboards. Expert in writing complex SQL queries, window functions, and CTEs across PostgreSQL and Snowflake warehouses. Designed automated interactive reporting suites in Power BI and Tableau. Proficient in advanced Excel financial modeling and basic Python exploratory analysis with Pandas. Excellent stakeholder communication and presentation skills.",
            "Commercial Data Analyst with demonstrated success improving operational KPI reporting. Built cross-functional Power BI dashboards connecting live database tables. Conducted cohort analyses, customer retention studies, and A/B testing result validations using SQL and Excel. Strong problem solving and cross-functional team leadership.",
            "Analytics Specialist skilled in data warehousing, dashboard design, and ETL hygiene. Automated daily reporting workflows using SQL and Tableau. Collaborated closely with product managers to deliver KPI metrics and cohort visualizations. Proficient in Excel modeling and statistical analysis."
        ]
    },
    "Python Backend Engineer": {
        "titles": ["Senior Python Backend Engineer", "Python API Developer", "Backend Systems Engineer", "Microservices Developer"],
        "core_skills": ["Python", "FastAPI", "PostgreSQL", "REST APIs", "Docker", "Git", "Unit Testing", "Microservices"],
        "extra_skills": ["Redis", "Apache Kafka", "Django", "AWS", "CI/CD", "System Design", "Linux"],
        "degrees": ["Bachelor's Degree", "Master's Degree"],
        "exp_range": (3, 8),
        "templates": [
            "High-performance Python Backend Engineer with 5+ years architecting microservices and asynchronous REST APIs using FastAPI and Django. Designed normalized relational schemas and optimized queries in PostgreSQL. Implemented sub-millisecond in-memory caching using Redis and asynchronous event streaming with Apache Kafka. Containerized services using Docker and maintained comprehensive unit tests with Pytest. Committed to Clean Code and Git workflows.",
            "Backend Systems Engineer specialized in distributed Python microservices and high-load web architectures. Built scalable RESTful endpoints with FastAPI, orchestrated background tasks, and secured endpoints with JWT auth. Utilized PostgreSQL for persistent storage, Redis for rate limiting, and Docker for containerized deployment on AWS.",
            "Python Software Engineer with extensive experience in REST API development, asynchronous programming, and relational database modeling. Developed robust web backends with FastAPI and Django, wrote automated unit tests with Pytest, and managed Docker containers in CI/CD pipelines."
        ]
    },
    "Junior / Generalist Developer": {
        "titles": ["Junior Software Developer", "Associate Web Programmer", "Entry-Level Programmer", "Computer Science Graduate"],
        "core_skills": ["Python", "JavaScript", "HTML5", "CSS3", "Git"],
        "extra_skills": ["SQL", "Flask", "Problem Solving", "Communication"],
        "degrees": ["Bachelor's Degree", "Associate / Diploma"],
        "exp_range": (0, 2),
        "templates": [
            "Recent Computer Science graduate with solid foundational coursework in data structures, algorithms, and software engineering. Proficient in Python and JavaScript, with hands-on academic projects building simple web apps using Flask, HTML5, and CSS3. Familiar with Git version control, basic SQL queries, and object-oriented programming. Eager to learn modern frameworks and contribute to a fast-paced development team.",
            "Aspiring Junior Developer passionate about web technologies and coding best practices. Built personal full-stack projects using JavaScript, HTML5, and Python. Completed internships working with Git collaboration and relational databases. Strong enthusiasm for continuous learning and problem solving."
        ]
    }
}

FIRST_NAMES = [
    "Alex", "Sarah", "David", "Emily", "Marcus", "Priya", "Carlos", "Elena",
    "Michael", "Jessica", "Rohan", "Ananya", "Liam", "Sophia", "Wei", "Amina",
    "Daniel", "Chloe", "Arjun", "Fatima", "Lucas", "Maya", "Kevin", "Hannah",
    "Siddharth", "Zoe", "Gabriel", "Olga", "Nathan", "Grace"
]

LAST_NAMES = [
    "Chen", "Jenkins", "Kumar", "Zhang", "Brooks", "Sharma", "Rodriguez", "Petrov",
    "Taylor", "Miller", "Patel", "Iyer", "Smith", "Johnson", "Wang", "Al-Mansoor",
    "O'Connor", "Davis", "Reddy", "Hassan", "Silva", "Lin", "Anderson", "Kim",
    "Verma", "Wilson", "Santos", "Novak", "Wright", "Hall"
]

UNIVERSITIES = [
    "Stanford University", "Carnegie Mellon University", "UC Berkeley", "MIT",
    "University of Washington", "Georgia Tech", "University of Illinois", "UT Austin",
    "Purdue University", "University of Michigan", "Columbia University", "Cornell University"
]


def generate_resume_record(candidate_id: int, domain_name: str, candidate_name: str = None) -> Dict:
    """Generates a structured candidate record with realistic resume text."""
    domain = DOMAINS[domain_name]

    if not candidate_name:
        candidate_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

    clean_email_name = candidate_name.lower().replace(" ", ".")
    email = f"{clean_email_name}@email.com"
    phone = f"+1 ({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"

    title = random.choice(domain["titles"])
    min_exp, max_exp = domain["exp_range"]
    exp_years = round(random.uniform(min_exp, max_exp), 1)
    degree = random.choice(domain["degrees"])
    university = random.choice(UNIVERSITIES)

    # Core skills + subset of extra skills
    sample_extra = random.sample(domain["extra_skills"], k=random.randint(3, len(domain["extra_skills"])))
    all_skills = list(dict.fromkeys(domain["core_skills"] + sample_extra))

    template = random.choice(domain["templates"])

    # Build rich resume text
    resume_text = f"""{candidate_name}
{title}
Contact: {email} | Phone: {phone}
Location: San Francisco, CA | LinkedIn: linkedin.com/in/{clean_email_name}

SUMMARY
{template} Experienced professional with {exp_years} years of demonstrated success across engineering and analytics environments.

TECHNICAL SKILLS
Technical Proficiencies: {', '.join(all_skills)}
Core Competencies: Software Architecture, Clean Code, Agile Sprints, System Reliability

PROFESSIONAL EXPERIENCE
{title} | Tech Enterprise Corp (2022 - Present)
- Engineered scalable systems and production-grade architectures utilizing {', '.join(all_skills[:4])}.
- Optimized performance, reduced latency by 35%, and spearheaded cross-functional delivery with product teams.
- Collaborated using Git version control and conducted peer code reviews in Agile sprint cycles.

Associate Specialist | Digital Innovations LLC (2019 - 2022)
- Implemented core features and data pipelines utilizing {', '.join(all_skills[2:6])}.
- Partnered with senior leadership to define technical roadmaps and automated deployment workflows.

EDUCATION
{degree} in Computer Science / Engineering
{university} (Graduated with Honors)
"""

    return {
        "candidate_id": f"CAND-{candidate_id:04d}",
        "candidate_name": candidate_name,
        "email": email,
        "phone": phone,
        "current_title": title,
        "experience_years": exp_years,
        "degree": degree,
        "primary_category": domain_name,
        "skills_list": ", ".join(all_skills),
        "resume_text": resume_text.strip()
    }


def generate_pdf_resume(candidate_data: Dict, output_path: str):
    """Generates a clean PDF resume from candidate data using ReportLab."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ResumeTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor="#1F2937",
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "ResumeSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        textColor="#4B5563",
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        "ResumeSection",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor="#2563EB",
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "ResumeBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor="#374151",
        spaceAfter=6
    )

    story = []

    # Header
    story.append(Paragraph(candidate_data["candidate_name"], title_style))
    contact_str = f"<b>{candidate_data['current_title']}</b> | {candidate_data['email']} | {candidate_data['phone']}"
    story.append(Paragraph(contact_str, subtitle_style))
    story.append(Spacer(1, 6))

    # Parse sections from resume_text
    lines = candidate_data["resume_text"].split("\n")
    current_sec = ""
    current_content = []

    for line in lines[3:]:
        line_s = line.strip()
        if not line_s:
            continue
        if line_s in ["SUMMARY", "TECHNICAL SKILLS", "PROFESSIONAL EXPERIENCE", "EDUCATION"]:
            if current_sec and current_content:
                story.append(Paragraph(current_sec, section_style))
                story.append(Paragraph("<br/>".join(current_content), body_style))
                story.append(Spacer(1, 4))
                current_content = []
            current_sec = line_s
        else:
            current_content.append(line_s)

    if current_sec and current_content:
        story.append(Paragraph(current_sec, section_style))
        story.append(Paragraph("<br/>".join(current_content), body_style))

    doc.build(story)


def generate_all_data(output_dir: str):
    """
    Main generator routine creating:
    - 120-row resume dataset CSV
    - Sample PDF and TXT resumes
    - TF-IDF vectorizer artifact
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_dir = os.path.join(output_dir, "sample_resumes")
    os.makedirs(sample_dir, exist_ok=True)

    records = []
    cand_id = 1

    # 1. Generate 6 curated benchmark candidates
    curated_profiles = [
        ("Alex Chen", "Machine Learning Engineer", "resume_alex_chen_ml"),
        ("Sarah Jenkins", "Full Stack Developer", "resume_sarah_jenkins_frontend"),
        ("David Kumar", "Cloud & DevOps Engineer", "resume_david_kumar_devops"),
        ("Emily Zhang", "Full Stack Developer", "resume_emily_zhang_fullstack"),
        ("Marcus Brooks", "Data Analyst", "resume_marcus_brooks_data_analyst"),
        ("Jordan Lee", "Junior / Generalist Developer", "resume_jordan_lee_junior_sw")
    ]

    for name, domain, base_filename in curated_profiles:
        rec = generate_resume_record(cand_id, domain, candidate_name=name)
        records.append(rec)
        cand_id += 1

        # Write TXT resume
        txt_path = os.path.join(sample_dir, f"{base_filename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(rec["resume_text"])

        # Write PDF resume
        pdf_path = os.path.join(sample_dir, f"{base_filename}.pdf")
        generate_pdf_resume(rec, pdf_path)

    # 2. Generate remaining realistic candidate pool (total 120 candidates)
    domain_keys = list(DOMAINS.keys())
    for _ in range(114):
        dom = random.choice(domain_keys)
        rec = generate_resume_record(cand_id, dom)
        records.append(rec)
        cand_id += 1

    # Save to CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "resumes_dataset.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df)} resumes to {csv_path}")

    # 3. Fit and save TF-IDF Vectorizer model artifact
    models_dir = os.path.join(os.path.dirname(output_dir), "models")
    os.makedirs(models_dir, exist_ok=True)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=6000,
        sublinear_tf=True
    )
    vectorizer.fit(df["resume_text"])
    vectorizer_path = os.path.join(models_dir, "resume_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)

    metadata = {
        "num_resumes": len(df),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "domains": list(DOMAINS.keys()),
        "model_version": "1.0.0",
        "created_at": "2026-11-01"
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved fitted vectorizer to {vectorizer_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    generate_all_data(data_dir)
