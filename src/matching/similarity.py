"""Compute semantic similarity between resumes and a job description.

Pipeline:
  1. Encode resume text + JD text with sentence-transformers
  2. Compute cosine similarity for each resume
  3. Return candidates ranked by score (descending)

Usage:
    from src.matching.similarity import rank_candidates

    candidates = [
        {"name": "alice.pdf", "text": "Alice is a Python developer …"},
        {"name": "bob.pdf",   "text": "Bob has 5 years in Java …"},
    ]
    jd_text = "We are looking for a Python backend engineer …"
    ranked = rank_candidates(candidates, jd_text)
    # → [{"name": "alice.pdf", "score": 0.87, "text": "…"}, …]
"""

from functools import lru_cache

from src.utils.helpers import get_logger, load_config

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load SentenceTransformer model once and cache it."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def rank_candidates(
    candidates: list[dict],
    jd_text: str,
    model_name: str | None = None,
    top_n: int | None = None,
) -> list[dict]:
    """Rank resume candidates against a job description by cosine similarity.

    Args:
        candidates: List of dicts with at least {"name": str, "text": str}.
        jd_text:    Job description plain text.
        model_name: SentenceTransformer model name (default from config.yaml).
        top_n:      Return only top N results (default from config.yaml).

    Returns:
        List of dicts sorted by score descending:
        [{"name": str, "score": float, "text": str}, …]
    """
    from sentence_transformers import util

    config = load_config()
    if model_name is None:
        model_name = config["matching"]["embedding_model"]
    if top_n is None:
        top_n = config["output"]["top_n"]
    threshold = config["matching"]["similarity_threshold"]

    if not candidates:
        logger.warning("rank_candidates() received empty candidate list")
        return []
    if not jd_text.strip():
        raise ValueError("jd_text must not be empty")

    model = _load_model(model_name)

    # Encode all texts in one batch (faster)
    resume_texts = [c["text"] for c in candidates]
    logger.info(f"Encoding {len(resume_texts)} resume(s) + JD …")

    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True, show_progress_bar=False)
    jd_embedding      = model.encode(jd_text,      convert_to_tensor=True, show_progress_bar=False)

    # Cosine similarity: shape (n_resumes,)
    scores = util.cos_sim(jd_embedding, resume_embeddings)[0].tolist()

    results = []
    for candidate, score in zip(candidates, scores):
        if score >= threshold:
            results.append({**candidate, "score": round(float(score), 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_n]

    logger.info(f"Ranked {len(results)} candidate(s) (threshold={threshold})")
    return results
