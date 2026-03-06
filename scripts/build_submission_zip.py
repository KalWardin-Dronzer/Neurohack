from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
ZIP_PATH = DIST_DIR / "long_form_memory_submission.zip"

ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    ".env.example",
    "run_demo.sh",
    "SUBMISSION.md",
    "ARCHITECTURE_DIAGRAM.txt",
    "API_SAMPLE_RESPONSES.md",
]

FOLDERS = ["app", "scripts", "tests"]

EXCLUDED_DIR_NAMES = {"__pycache__", ".pytest_cache", ".venv", "dist", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pyd", ".db", ".db-shm", ".db-wal"}


def include_path(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def build_zip() -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with ZipFile(ZIP_PATH, mode="w", compression=ZIP_DEFLATED) as archive:
        for name in ROOT_FILES:
            file_path = ROOT / name
            if file_path.exists() and file_path.is_file():
                archive.write(file_path, file_path.relative_to(ROOT))

        for folder_name in FOLDERS:
            folder_path = ROOT / folder_name
            if not folder_path.exists():
                continue
            for file_path in folder_path.rglob("*"):
                if not file_path.is_file():
                    continue
                if not include_path(file_path.relative_to(ROOT)):
                    continue
                archive.write(file_path, file_path.relative_to(ROOT))

    return ZIP_PATH


def main() -> None:
    zip_path = build_zip()
    print(f"Created: {zip_path}")


if __name__ == "__main__":
    main()
