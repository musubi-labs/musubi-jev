"""Where Kev lives on the Hugging Face Hub and GitHub. Code and scripts import these instead of spelling the ids out.
Changing the home means changing this file (and the pinned SUITES_REVISION in kev/suite.py)."""

ORG = "jaredpalmer"
GITHUB_REPO = f"{ORG}/kev"
SPACE_REPO = f"{ORG}/kev"
SUITES_REPO = f"{ORG}/kev-suites"


def model_repo(size: str) -> str:
    """Hub id of a released checkpoint, e.g. model_repo("4b") -> "jaredpalmer/kev-4b"."""
    return f"{ORG}/kev-{size}"
