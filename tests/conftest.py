import sys
from pathlib import Path


def pytest_configure(config):
    """Ensure project root is on sys.path for tests.

    This makes imports like `from app.main import app` work when running pytest
    from the repository root.
    """
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
