# 📄 Resume Screening System

Automated resume screening pipeline using NER and semantic similarity —
rank candidates against a job description in seconds.

## Overview

Upload resumes (PDF / DOCX / CSV), paste or upload a job description,
and the system automatically extracts skills, companies, and dates using
a two-layer NER pipeline (spaCy + HuggingFace), then ranks all candidates
by semantic similarity using sentence-transformers. Results are displayed
in an interactive Streamlit dashboard with ranking table, entity details,
and score charts.

## Live Demo

> 🚀 *(Streamlit Community Cloud link — add after deploy)*

## Features

| Feature | Detail |
|---|---|
| 📂 File support | PDF, DOCX, CSV, TXT (up to 10 files) |
| 🔍 NER Layer 1 | spaCy `en_core_web_sm` → PERSON, ORG, DATE, GPE |
| 🛠 NER Layer 2 | `algiraldohe/lm-ner-linkedin-skills-recognition` → SKILL |
| 📐 Similarity | `all-MiniLM-L6-v2` cosine similarity |
| 📊 UI | Streamlit — Ranking, Entities, Chart tabs |
| 💾 Demo mode | 10 pre-built sample candidates auto-loaded on open |

## Project Structure

```
llm_resume_screening/
├── app.py                  # Streamlit entrypoint
├── configs/config.yaml     # model names, thresholds
├── src/
│   ├── parsers/            # PDF/DOCX/CSV → plain text
│   ├── ner/                # spaCy + HF NER extraction
│   └── matching/           # sentence-transformers ranking
├── data/raw/               # sample resumes + job description
├── notebooks/              # NER & similarity experiments
├── assets/profile.png      # profile photo (not tracked by git)
└── requirements.txt
```

## Setup

```powershell
# 1. Create virtual environment (uv recommended)
uv venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Run the app
streamlit run app.py
```

> HuggingFace models (`algiraldohe/lm-ner-linkedin-skills-recognition`,
> `all-MiniLM-L6-v2`) download automatically on first run (~400 MB)
> and are cached at `~/.cache/huggingface/hub/`.

## Usage

**Demo mode** — open the app and the Ranking tab shows pre-screened
results for 10 sample candidates against a Senior Data Scientist / NLP
Engineer job description.

**Upload mode:**
1. Go to **📝 Job Description** tab → paste or upload JD
2. In the sidebar → select "Upload Files" → drop resumes
3. Click **🚀 Run Screening**
4. View results in **📊 Ranking & Stats**, **🔍 Entities**, **📈 Chart** tabs

## Tech Stack

- Python 3.13
- [spaCy](https://spacy.io/) 3.8 — general NER
- [sentence-transformers](https://www.sbert.net/) 5.5 — semantic similarity
- [HuggingFace Transformers](https://huggingface.co/) 5.9 — skill NER
- [Streamlit](https://streamlit.io/) 1.57 — web UI
- PyPDF2, python-docx — file parsing
- Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)

## Author

👤 [Yuttapong M.](https://www.linkedin.com/in/yuttapong-m/)
