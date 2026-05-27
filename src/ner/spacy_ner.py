"""Layer 1 NER — spaCy: extracts PERSON, ORG, DATE, GPE from resume text."""

from functools import lru_cache

from src.utils.helpers import get_logger, load_config

logger = get_logger(__name__)

# Entity labels we care about from spaCy
SPACY_LABELS = {"PERSON", "ORG", "DATE", "GPE"}


@lru_cache(maxsize=1)
def _load_nlp(model_name: str):
    """Load spaCy model once and cache it."""
    try:
        import spacy
        logger.info(f"Loading spaCy model: {model_name}")
        return spacy.load(model_name)
    except OSError:
        raise OSError(
            f"spaCy model '{model_name}' not found.\n"
            f"Run: python -m spacy download {model_name}"
        )


def extract_general(text: str, model_name: str | None = None) -> dict:
    """Extract general entities from text using spaCy.

    Returns:
        {
            "persons":   ["Alice Chen"],
            "companies": ["Google", "OpenAI"],
            "dates":     ["2020–2023", "January 2019"],
            "locations": ["Bangkok", "New York"],
        }
    """
    if model_name is None:
        config = load_config()
        model_name = config["ner"]["spacy_model"]

    nlp = _load_nlp(model_name)
    doc = nlp(text)

    result: dict[str, list[str]] = {
        "persons":   [],
        "companies": [],
        "dates":     [],
        "locations": [],
    }

    seen: set[str] = set()
    for ent in doc.ents:
        val = ent.text.strip()
        if ent.label_ not in SPACY_LABELS or val in seen:
            continue
        seen.add(val)
        if ent.label_ == "PERSON":
            result["persons"].append(val)
        elif ent.label_ == "ORG":
            result["companies"].append(val)
        elif ent.label_ == "DATE":
            result["dates"].append(val)
        elif ent.label_ == "GPE":
            result["locations"].append(val)

    logger.info(
        f"[spaCy] persons={len(result['persons'])} "
        f"companies={len(result['companies'])} "
        f"dates={len(result['dates'])} "
        f"locations={len(result['locations'])}"
    )
    return result
