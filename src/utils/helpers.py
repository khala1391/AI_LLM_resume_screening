"""Shared utilities: config loader, logging setup, file helpers."""

import logging
import yaml
from pathlib import Path


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path | None = None) -> dict:
    """Load configs/config.yaml (or a custom path) and return as dict."""
    if config_path is None:
        # default: project_root/configs/config.yaml
        config_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── File I/O ──────────────────────────────────────────────────────────────────

def list_files(folder: str | Path, extensions: list[str]) -> list[Path]:
    """Return all files in folder matching given extensions (e.g. ['.pdf', '.docx'])."""
    folder = Path(folder)
    return [p for p in folder.iterdir() if p.suffix.lower() in extensions]


def save_text(text: str, dest: str | Path) -> None:
    """Write plain text to dest, creating parent dirs as needed."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
