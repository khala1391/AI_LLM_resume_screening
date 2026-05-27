# Resume Screening System — Task Prompt

## Context
A resume screening pipeline that accepts resume files (PDF/DOCX), extracts
structured information (skills, experience, education) using Named Entity
Recognition (NER), then uses `sentence-transformers` to compute semantic
similarity between each resume and a given job description. Candidates are
ranked by similarity score and presented via a Streamlit web app.

Project folder structure has already been created:
  llm_resume_screening/
  ├── data/{raw, interim, processed}/
  ├── notebooks/
  ├── src/{parsers, ner, matching, utils}/
  ├── models/
  ├── configs/
  ├── outputs/
  ├── app.py              ← Streamlit entrypoint (to be created)
  └── requirements.txt

## Goal
- Parse uploaded resume files (PDF / DOCX) into plain text
- Extract entities using a **3-layer NER approach**:
  - **Layer 1 — spaCy `en_core_web_sm`**: extract `PERSON`, `ORG` (companies),
    `DATE` (employment periods), `GPE` (locations)
  - **Layer 2 — HuggingFace `algiraldohe/lm-ner-linkedin-skills-recognition`**:
    extract `SKILL` entities from resume text
  - **Layer 3 — Rule-based `EntityRuler`**: skip (not needed for this project)
  - Merge results from Layer 1 + Layer 2 into a single structured dict
- Encode extracted resume text and a user-provided job description using
  `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
- Compute cosine similarity scores between each resume and the JD
- Rank applicants by score and display results in a Streamlit UI

## Deliverables
- [ ] `.venv` with all dependencies installed
- [ ] `notebooks/01_explore_ner.ipynb` — prototype spaCy NER extraction
      on sample resumes, inspect entity labels
- [ ] `notebooks/02_similarity_tuning.ipynb` — experiment with
      sentence-transformer models, compare similarity scores
- [ ] `src/parsers/resume_parser.py` — PDF/DOCX → plain text
- [ ] `src/parsers/jd_parser.py` — job description text loader
- [ ] `src/ner/extractor.py` — spaCy NER wrapper; returns dict of
      {skills, experience, education}
- [ ] `src/matching/similarity.py` — encode texts, compute cosine
      similarity, return ranked list
- [ ] `src/utils/helpers.py` — file I/O, logging, config loader
- [ ] `configs/config.yaml` — model names, NER label map, score thresholds
- [ ] `app.py` — Streamlit UI: upload resumes, paste JD, display ranking table
- [ ] `README.md` — project overview, setup steps, how to run

## Data
- Input resumes: placed in `data/raw/` (PDF or DOCX format)
- Job descriptions: text string (entered in UI) or `.txt` files in `data/raw/`
- Interim parsed text: `data/interim/`
- Processed embeddings/entities: `data/processed/`
- No labelled training data needed (using pre-trained spaCy + sentence-transformers)

## Constraints / Preferences
- Python 3.10+
- Key libraries: `spacy`, `sentence-transformers`, `pypdf2`, `python-docx`,
  `streamlit`, `pandas`, `pyyaml`, `tqdm`
- NER model: `en_core_web_sm` (or `en_core_web_trf` for better accuracy)
- Embedding model: `all-MiniLM-L6-v2` (fast, good quality) — configurable
  via `configs/config.yaml`
- Deployment: Streamlit Community Cloud (requires public GitHub repo +
  `requirements.txt` at root)

## Acceptance Criteria
- Given ≥ 2 resume files and a job description, the pipeline ranks all
  resumes with a similarity score between 0–1
- The Streamlit app runs locally with `streamlit run app.py` without errors
- Extracted entities (skills, experience, education) are visible in the UI
  or a debug panel
- App is deployable on Streamlit Community Cloud from GitHub
