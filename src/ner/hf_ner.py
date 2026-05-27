"""Layer 2 NER — HuggingFace: extracts SKILL entities from resume text.

Model: algiraldohe/lm-ner-linkedin-skills-recognition
Weights are downloaded automatically on first use (~400MB) and cached at
~/.cache/huggingface/hub/
"""

from functools import lru_cache

from src.utils.helpers import get_logger, load_config

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_pipeline(model_name: str):
    """Load HuggingFace NER pipeline once and cache it."""
    from transformers import pipeline
    logger.info(f"Loading HF NER model: {model_name} (downloads on first use)")
    return pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
    )


def extract_skills(text: str, model_name: str | None = None) -> dict:
    """Extract skill entities from text using the LinkedIn skills NER model.

    Returns:
        {
            "skills": ["Python", "Machine Learning", "Docker", ...],
        }
    """
    if model_name is None:
        config = load_config()
        model_name = config["ner"]["hf_skills_model"]

    ner_pipeline = _load_pipeline(model_name)
    raw_entities = ner_pipeline(text)

    # Deduplicate, normalise case
    seen: set[str] = set()
    skills: list[str] = []
    for ent in raw_entities:
        word = ent["word"].strip()
        word_lower = word.lower()
        if word_lower not in seen and len(word) > 1:
            seen.add(word_lower)
            skills.append(word)

    logger.info(f"[HF NER] skills found: {len(skills)}")
    return {"skills": skills}
