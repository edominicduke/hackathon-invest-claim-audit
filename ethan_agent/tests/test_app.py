"""Drive the Streamlit UI headlessly through the full flow with the scripted LLM."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import agent
from agent.pipeline import ThesisAuditor
from agent.stubs import FixtureRetriever, JsonFileStore
from tests.test_pipeline import make_llm

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def button(at, label):
    return next(b for b in at.button if b.label == label)


def page_html(at) -> str:
    return " ".join(m.value for m in at.markdown)


def start(monkeypatch, tmp_path):
    fake = ThesisAuditor(make_llm(), FixtureRetriever(agent.FIXTURES), store=JsonFileStore(tmp_path), max_workers=1)
    monkeypatch.setattr(agent, "build_default_auditor", lambda **_: fake)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    return at


def run_audit(at):
    button(at, "Check my thesis").click().run()
    assert at.text_input[0].value == "EXMP"           # "confirm the details" panel
    button(at, "Confirm and run audit").click().run()
    assert not at.exception
    return at


def test_full_ui_flow(monkeypatch, tmp_path):
    at = run_audit(start(monkeypatch, tmp_path))
    page = page_html(at)
    assert "Thesis well supported" in page
    assert 'class="cite"' in page                      # [S#] rendered as citation chips
    assert any("no real competition" in t.label for t in at.text_area)  # clarifying question shown

    answer_box = next(t for t in at.text_area if "no real competition" in t.label)
    answer_box.input("I'd sell if growth fell below 30%")
    button(at, "Save my answers").click().run()
    assert not at.exception
    assert list((tmp_path / "answers").glob("*.json"))


def test_example_buttons_fill_the_input(monkeypatch, tmp_path):
    at = start(monkeypatch, tmp_path)
    button(at, "Weak thesis").click().run()
    assert "gaming business" in at.text_area[0].value


def test_analyst_view_shows_audit_trail(monkeypatch, tmp_path):
    at = start(monkeypatch, tmp_path)
    at.session_state["view"] = "Analyst"
    run_audit(at)
    assert not at.exception
    assert any(e.label.startswith("Audit trail") for e in at.expander)
    assert any(b.label == "Download audit (JSON)" for b in at.get("download_button")) or "Audit record" in page_html(at)


def test_model_text_is_escaped(monkeypatch, tmp_path):
    """Text from documents/model must never render as live HTML."""
    at = run_audit(start(monkeypatch, tmp_path))
    audit = at.session_state["audit"]
    audit.claims[0].reasoning = '<img src=x onerror="alert(1)"> revenue grew [S1].'
    at.run()
    page = page_html(at)
    assert "<img src=x" not in page and "&lt;img src=x" in page
