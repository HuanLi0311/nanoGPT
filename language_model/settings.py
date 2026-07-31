"""Load the single YAML file that defines a reproducible run."""

from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as error:
    raise SystemExit("Missing dependency. Run: pip install PyYAML") from error


def load_settings(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Settings file does not exist: {path}")
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a YAML mapping.")
    return settings
