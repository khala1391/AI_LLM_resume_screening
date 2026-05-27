"""Main NER extractor — merges Layer 1 (spaCy) + Layer 2 (HF skills).

Usage:
    from src.ner.extractor import extract
    result = extract("Alice is a Python developer at Google since 2020.")
    # → {
    #     "persons":   ["Alice"],
    #     "companies": ["Google"],
    #     "dates":     ["2020"],
    #     "locations": [],
    #     "skills":    ["Python"],
    # }
"""

from src.ner.spacy_ner import extract_general
from src.ner.hf_ner import extract_skills
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def extract(text: str) -> dict:
    """Run all NER layers and return a merged entity dict.

    Args:
        text: Plain text from a resume.

    Returns:
        {
            "persons":   list[str],   # candidate names
            "companies": list[str],   # employers / organisations
            "dates":     list[str],   # employment periods / graduation years
            "locations": list[str],   # cities / countries
            "skills":    list[str],   # technical & soft skills
        }
    """
    if not text or not text.strip():
        logger.warning("extract() received empty text — returning empty result")
        return {
            "persons": [], "companies": [],
            "dates": [], "locations": [], "skills": []
        }

    logger.info("Running NER extraction …")

    # Layer 1 — spaCy: PERSON, ORG, DATE, GPE
    general = extract_general(text)

    # Layer 2 — HuggingFace: SKILL
    skills = extract_skills(text)

    merged = {**general, **skills}

    logger.info(
        f"Extraction complete → "
        f"skills={len(merged['skills'])} | "
        f"companies={len(merged['companies'])} | "
        f"dates={len(merged['dates'])}"
    )
    return merged
