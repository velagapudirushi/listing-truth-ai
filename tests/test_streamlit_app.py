"""Headless Streamlit AppTest verifying all 4 tabs, KPI metrics, and SQL execution."""

from pathlib import Path
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app" / "streamlit_app.py"


def test_streamlit_multimodal_app_renders_cleanly() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()

    assert len(at.exception) == 0, f"Unexpected UI exception: {at.exception}"
    assert len(at.error) == 0, f"Unexpected UI error: {at.error}"
    assert len(at.tabs) == 4
    assert len(at.metric) >= 4

    # Click 'Execute DuckDB Query' button in Tab 4
    at.button[0].click().run()
    assert len(at.exception) == 0
