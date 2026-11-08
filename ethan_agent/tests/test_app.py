"""Drive the Streamlit UI headlessly through the full flow with the scripted LLM."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import agent
from agent.pipeline import ThesisAuditor
from agent.stubs import FixtureRetriever, JsonFileStore
from tests.test_pipeline import make_llm

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def fresh_auditor_cache():
    """app.py caches the auditor (st.cache_resource), and that cache outlives a single
    AppTest run; without this, a test can silently get the previous test's auditor."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


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


THESIS = "I'm buying EXMP because AI demand will keep data center revenue growing 50%+ and they have no real competition."


def run_audit(at):
    at.text_area[0].set_value(THESIS).run()           # the box starts empty (placeholder only)
    button(at, "Check my thesis").click().run()
    assert at.text_input[0].value == "EXMP"           # "confirm the details" panel
    button(at, "Confirm and run audit").click().run()
    assert not at.exception
    return at


def test_full_ui_flow(monkeypatch, tmp_path):
    at = run_audit(start(monkeypatch, tmp_path))
    page = page_html(at)
    assert "Strong evidence" in page and "Evidence strength" in page
    assert 'class="cite"' in page                      # [S#] rendered as citation chips
    assert any("no real competition" in t.label for t in at.text_area)  # clarifying question shown

    answer_box = next(t for t in at.text_area if "no real competition" in t.label)
    answer_box.input("I'd sell if growth fell below 30%")
    button(at, "Save my answers").click().run()
    assert not at.exception
    assert list((tmp_path / "answers").glob("*.json"))


def test_example_buttons_fill_the_input(monkeypatch, tmp_path):
    at = start(monkeypatch, tmp_path)
    assert at.text_area[0].value == ""                 # placeholder, not prefilled text
    assert button(at, "Check my thesis").disabled
    next(g for g in at.get("button_group") if g.label == "Examples").set_value("EXMP · Gaming +20%").run()
    assert not at.exception
    assert "gaming business" in at.text_area[0].value


def test_analyst_view_shows_audit_trail(monkeypatch, tmp_path):
    at = start(monkeypatch, tmp_path)
    at.session_state["view"] = "Analyst"
    run_audit(at)
    assert not at.exception
    assert any(e.label.startswith("Audit trail") for e in at.expander)
    assert any(b.label == "Download audit (JSON)" for b in at.get("download_button")) or "Audit record" in page_html(at)


def test_track_record_page(monkeypatch, tmp_path):
    from agent.integrations import StockAgentStore
    from tests.test_integrations import fake_prices

    store = StockAgentStore(tmp_path / "track", archive=JsonFileStore(tmp_path / "archive"), record_price=False)
    fake = ThesisAuditor(make_llm(), FixtureRetriever(agent.FIXTURES), store=store, max_workers=1)
    monkeypatch.setattr(agent, "build_default_auditor", lambda **_: fake)
    monkeypatch.setattr(store, "update_results", lambda prices=None: StockAgentStore.update_results(store, fake_prices()))
    at = AppTest.from_file(APP, default_timeout=30).run()
    run_audit(at)                                     # saved to the (temporary) track record
    at.switch_page("views/track_record.py").run()     # top navigation
    assert not at.exception
    page = page_html(at)
    assert "Track record" in page and "Audits saved" in page
    assert at.dataframe and any(e.label.startswith(tuple("0123456789")) and "EXMP" in e.label for e in at.expander)
    button(at, "Check results now").click().run()
    assert not at.exception
    assert any("Checked 1 audits" in c.value for c in at.caption)


def test_track_record_without_team_storage(monkeypatch, tmp_path):
    at = start(monkeypatch, tmp_path)
    at.switch_page("views/track_record.py").run()
    assert not at.exception
    assert any("isn't loaded" in i.value for i in at.info)


def test_methodology_page_and_no_sidebar(monkeypatch, tmp_path):
    at = start(monkeypatch, tmp_path)
    assert len(at.sidebar) == 0                       # the app no longer uses a sidebar
    at.switch_page("views/methodology.py").run()
    assert not at.exception
    assert "Methodology" in page_html(at)


def test_view_toggle_lives_in_results(monkeypatch, tmp_path):
    at = start(monkeypatch, tmp_path)
    assert not any(s.label == "View" for s in at.get("button_group"))   # not shown before an audit
    run_audit(at)
    toggle = next(s for s in at.get("button_group") if s.label == "View")
    toggle.set_value("Analyst").run()
    assert not at.exception
    assert any(e.label.startswith("Audit trail") for e in at.expander)  # analyst-only detail appears


def test_model_text_is_escaped(monkeypatch, tmp_path):
    """Text from documents/model must never render as live HTML."""
    at = run_audit(start(monkeypatch, tmp_path))
    audit = at.session_state["audit"]
    audit.claims[0].reasoning = '<img src=x onerror="alert(1)"> revenue grew [S1].'
    at.run()
    page = page_html(at)
    assert "<img src=x" not in page and "&lt;img src=x" in page
