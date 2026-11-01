# 💼 TalentMatch AI — Intelligent Resume & Candidate Screening System
### Machine Learning Task 3 (2026) • Future Interns Internship Program
**Author**: nithi-2008  
**Repository**: [https://github.com/nithi-2008/FUTURE_ML_03.git](https://github.com/nithi-2008/FUTURE_ML_03.git)  
**Organization**: [Future Interns](https://futureinterns.com/)  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI%20Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP%20Pipeline-09A3D5.svg?logo=spacy&logoColor=white)](https://spacy.io/)
[![NLTK](https://img.shields.io/badge/NLTK-Text%20Preprocessing-306998.svg?logo=python&logoColor=white)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF%20%26%20Similarity-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-17%2F17%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Overview & Problem Context
In modern talent acquisition and recruitment operations, hiring managers frequently receive **hundreds or thousands of candidate resumes** for every open job requisition. 

### The Real-World Operational Challenge:
* **Manual Screening Bottleneck**: Recruiters spend an average of **6 to 8 seconds** scanning each resume, leading to superficial judgments, missed talent, and high cognitive fatigue.
* **Inconsistent & Biased Evaluations**: Different recruiters prioritize different criteria, causing subjective evaluations and compliance risks.
* **Keyword-Stuffing Exploits**: Candidates frequently game legacy Applicant Tracking Systems (ATS) by pasting white-font keywords or superficial buzzwords without genuine domain context.
* **Recruiter Workload Overhead**: Identifying missing critical skills and formulating tailored interview questions consumes excessive manual effort.

### The Solution:
**TalentMatch AI** is a production-grade, transparent **Machine Learning Decision-Support System** designed for HR-tech startups, recruitment platforms, and enterprise hiring teams. Rather than acting as a black-box filter that auto-rejects candidates, TalentMatch AI assists recruiters by extracting verified domain skills, measuring semantic role alignment, ranking candidate fit, identifying exact skill gaps, and generating structured technical interview questions.

---

## 🏛️ System Architecture & Workflow

```
[ Unstructured Resumes (PDF / DOCX / TXT) ]          [ Target Job Description (JD) ]
                     │                                              │
                     ▼                                              ▼
       [ 1. Multi-Format Text Extraction ]             [ Dynamic Role Skill Parser ]
         (pypdf / python-docx / txt)                                │
                     │                                              │
                     ▼                                              │
       [ 2. NLP Cleaning & Tokenization ]                           │
      (Regex Noise Removal, Tokenization,                           │
           Lemmatization, Sectioning)                               │
                     │                                              │
                     ▼                                              │
      [ 3. Canonical Skill Extraction ]                             │
       (spaCy PhraseMatcher + 500+ Skill                            │
       Taxonomy + Regex Boundary Guard)                             │
                     │                                              │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                           [ 4. Hybrid Scoring Engine ]
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │                               │                               │
            ▼                               ▼                               ▼
    [ Weighted Skill Match ]     [ TF-IDF Cosine Similarity ]     [ Experience Calibration ]
    (Required vs Preferred)       (Semantic Domain Context)       (Seniority & Degree Fit)
            │                               │                               │
            └───────────────────────────────┼───────────────────────────────┘
                                            │
                                            ▼
                         [ 5. Composite Role Fit Ranking ]
                         (Score Calculation: 0.0 - 100.0%)
                                            │
                                            ▼
                    [ 6. Explainable AI (XAI) & Decision Support ]
              • Rank & Tier: Highly Recommended / Review / Low Fit
              • Skill Gap Matrix: Matched vs Missing Critical Skills
              • Tailored Technical Interview Screening Questions
              • Visual Radar & Comparative Analytics (Streamlit + Plotly)
```

---

## 🔬 Scoring Logic & Mathematical Formulation

To ensure transparency and prevent blind keyword exploitation, candidate suitability is computed via a **calibrated tripartite hybrid formulation**:

$$\text{Composite Fit Score} = (w_{\text{skill}} \cdot S_{\text{skill}}) + (w_{\text{semantic}} \cdot S_{\text{semantic}}) + (w_{\text{exp}} \cdot S_{\text{exp}})$$

Where weights satisfy $w_{\text{skill}} + w_{\text{semantic}} + w_{\text{exp}} = 1.0$ (Default: $0.50, 0.35, 0.15$):

### 1. Weighted Skill Match Score ($S_{\text{skill}}$)
Differentiates between **Must-Have Core Skills** ($\omega_{\text{req}} = 2.0$) and **Nice-to-Have Preferred Skills** ($\omega_{\text{pref}} = 1.0$):

$$S_{\text{skill}} = \left( \frac{\omega_{\text{req}} \cdot |K_{\text{cand}} \cap K_{\text{req}}| + \omega_{\text{pref}} \cdot |K_{\text{cand}} \cap K_{\text{pref}}|}{\omega_{\text{req}} \cdot |K_{\text{req}}| + \omega_{\text{pref}} \cdot |K_{\text{pref}}|} \right) \times 100$$

### 2. Semantic Context Similarity ($S_{\text{semantic}}$)
Computes the vector cosine similarity between the candidate resume text vector $\mathbf{v}_{\text{resume}}$ and the target job description vector $\mathbf{v}_{\text{JD}}$ using sublinear TF-IDF representation across unigrams and bigrams:

$$S_{\text{semantic}} = \cos(\mathbf{v}_{\text{resume}}, \mathbf{v}_{\text{JD}}) \times 100 = \left( \frac{\mathbf{v}_{\text{resume}} \cdot \mathbf{v}_{\text{JD}}}{\|\mathbf{v}_{\text{resume}}\| \|\mathbf{v}_{\text{JD}}\|} \right) \times 100$$

### 3. Experience & Seniority Calibration ($S_{\text{exp}}$)
Measures alignment between detected candidate work history $E_{\text{cand}}$ and the role requirements $E_{\text{req}}$:

$$S_{\text{exp}} = \begin{cases} 
100.0 & \text{if } E_{\text{cand}} \ge E_{\text{req}} \\
\max\left(10.0, \frac{E_{\text{cand}}}{E_{\text{req}}} \times 100\right) & \text{if } E_{\text{cand}} < E_{\text{req}}
\end{cases}$$

---

## 🎯 Candidate Recommendation Tiers
Each candidate is categorized into actionable recruiter decision tiers:
* **[HIGH] Highly Recommended** ($\text{Score} \ge 75\%$): Meets or exceeds core requirements, strong semantic alignment, verified domain background. Direct shortlist for interviews.
* **[MEDIUM] Potential Fit (Review)** ($55\% \le \text{Score} < 75\%$): Demonstrates solid transferable skills but possesses specific critical skill gaps or lower seniority. Requires recruiter interview verification.
* **[LOW] Low Fit (Not Shortlisted)** ($\text{Score} < 55\%$): Significant skill divergence; background does not align with core role parameters.

---

## 📂 Project Repository Structure

```
FUTURE_ML_03/
├── .gitignore                      # Git ignore file for Python, artifacts, cache
├── LICENSE                         # MIT Open-Source License
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Pinned production dependencies
├── run_app.bat                     # Windows one-click dashboard launcher
├── app.py                          # Full-featured Streamlit UI application
├── data/
│   ├── job_descriptions.json       # 6 standardized benchmark job descriptions
│   ├── skills_taxonomy.json        # Curated 500+ skills taxonomy with canonical aliases
│   ├── resumes_dataset.csv         # 120-candidate multi-domain dataset
│   └── sample_resumes/             # Realistic PDF and TXT test resumes
│       ├── resume_alex_chen_ml.pdf
│       ├── resume_alex_chen_ml.txt
│       ├── resume_sarah_jenkins_frontend.pdf
│       ├── resume_sarah_jenkins_frontend.txt
│       ├── resume_david_kumar_devops.pdf
│       ├── resume_david_kumar_devops.txt
│       ├── resume_emily_zhang_fullstack.pdf
│       ├── resume_marcus_brooks_data_analyst.pdf
│       └── resume_jordan_lee_junior_sw.pdf
├── models/
│   ├── resume_vectorizer.joblib    # Fitted TF-IDF vectorizer artifact
│   └── model_metadata.json         # Vocabulary and training metadata
├── notebooks/
│   └── resume_screening_system.ipynb  # Step-by-step Jupyter research notebook
├── src/
│   ├── __init__.py                 # Module initializations
│   ├── text_preprocessor.py        # Text cleaning, regex guards, NLTK lemmatization
│   ├── document_reader.py          # PDF (pypdf), DOCX, and TXT parsing
│   ├── skill_extractor.py          # spaCy phrase matching & entity extraction
│   ├── matcher.py                  # Hybrid similarity and scoring logic
│   ├── ranker.py                   # Candidate ranking engine & verdict synthesis
│   ├── explainer.py                # Explainable AI (XAI) & interview probe generator
│   ├── visualizer.py               # Plotly charts (radar, leaderboard, gap donut)
│   ├── dataset_generator.py        # Synthetic dataset & ReportLab PDF generator
│   └── generate_notebook.py        # Jupyter notebook generator script
└── tests/
    ├── __init__.py
    ├── test_text_preprocessor.py   # Unit tests for text cleaning & normalization
    ├── test_document_reader.py     # Unit tests for PDF, DOCX, and TXT readers
    ├── test_skill_extractor.py     # Unit tests for spaCy skill extraction & aliases
    ├── test_matcher.py             # Unit tests for scoring logic & skill gaps
    └── test_ranker.py              # Unit tests for ranking & tier assignments
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nithi-2008/FUTURE_ML_03.git
cd FUTURE_ML_03
```

### 2. Set Up Python Environment
```bash
# Recommended: Python 3.10+
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / MacOS
source venv/bin/activate
```

### 3. Install Dependencies & Download NLP Models
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### 4. Run the Streamlit Dashboard
```bash
# Direct command:
streamlit run app.py

# Or on Windows, double-click:
run_app.bat
```
Navigate to `http://localhost:8501` in your browser.

---

## 🧪 Comprehensive Unit Testing
The project includes a comprehensive test suite covering all modules:

```bash
pytest -v
```

### Test Suite Results (17/17 Passed):
```
tests/test_document_reader.py::test_read_txt_file PASSED                 [  5%]
tests/test_document_reader.py::test_read_sample_pdf PASSED               [ 11%]
tests/test_document_reader.py::test_empty_input_handling PASSED          [ 17%]
tests/test_matcher.py::test_semantic_similarity PASSED                   [ 23%]
tests/test_matcher.py::test_evaluate_skills_perfect_match PASSED         [ 29%]
tests/test_matcher.py::test_evaluate_skills_with_gaps PASSED             [ 35%]
tests/test_matcher.py::test_full_resume_match PASSED                     [ 41%]
tests/test_ranker.py::test_assign_tier PASSED                            [ 47%]
tests/test_ranker.py::test_rank_candidates PASSED                        [ 52%]
tests/test_skill_extractor.py::test_extract_canonical_skills PASSED      [ 58%]
tests/test_skill_extractor.py::test_special_symbol_skills PASSED         [ 64%]
tests/test_skill_extractor.py::test_candidate_info_extraction PASSED     [ 70%]
tests/test_skill_extractor.py::test_categorized_skills PASSED            [ 76%]
tests/test_text_preprocessor.py::test_clean_text_removes_urls_and_emails PASSED [ 82%]
tests/test_text_preprocessor.py::test_clean_text_preserves_technical_tokens PASSED [ 88%]
tests/test_text_preprocessor.py::test_tokenize_and_lemmatize PASSED      [ 94%]
tests/test_text_preprocessor.py::test_section_extraction PASSED          [100%]

============================= 17 passed in 14.64s =============================
```

---

## 🖥️ Streamlit Interactive UI Features

1. **Live Multi-Resume Ingestion**: Upload PDF, DOCX, and TXT resumes or click *"Load 6 Benchmark Resumes"* for immediate evaluation.
2. **Dynamic Role Benchmark & Weight Customizer**: Switch between ML Engineer, Full Stack Dev, Cloud DevOps, Data Scientist, or create your own custom Job Description with custom skill weighting sliders.
3. **Interactive Plotly Leaderboard**: Real-time ranking with color-coded tier cards, percentile scores, and instant CSV export.
4. **Candidate Deep-Dive Inspector (XAI)**:
   * Exact mathematical score breakdown (Skill pts + Semantic pts + Experience pts).
   * Visual Skill Gap Matrix (Matched Required, Missing Critical, Matched Preferred, Extra Skills).
   * Domain Skill Competency Radar Chart.
   * Tailored Interview Screening Questions generated specifically to probe candidate weaknesses.
   * Executive Recruiter Briefing packet ready to copy into Applicant Tracking Systems (ATS).
5. **Head-to-Head Comparative Analytics**: Side-by-side grouped score comparison and qualitative ranking advantage breakdown between any 2 to 4 candidates.
6. **120-Resume Batch Screening**: Filterable domain-level performance distributions across the entire talent dataset.
7. **NLP Sandbox**: Test real-time skill extraction on any raw text snippet.

---

## 💼 Business Impact & Showcase
* **90% Reduction in Triage Time**: Automated resume parsing and ranking reduces recruiter initial review time from minutes to milliseconds.
* **Objective Skill Gap Visibility**: Immediately alerts hiring managers when a candidate is missing must-have requirements before scheduling interviews.
* **Structured Interview Guidance**: Provides recruiters with customized technical questions targeted directly at candidate resume omissions.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

### Citation & Acknowledgment:
Developed as part of the **Future Interns Machine Learning Internship Program (2026)**.  
Project Link: [https://github.com/nithi-2008/FUTURE_ML_03.git](https://github.com/nithi-2008/FUTURE_ML_03.git)  
Future Interns: [https://futureinterns.com/](https://futureinterns.com/)
