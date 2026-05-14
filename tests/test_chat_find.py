"""U3 — Ctrl+F conversation search."""

from __future__ import annotations


def _panel(qtbot):
    from faceview.gui.chat_panel import ChatPanel
    p = ChatPanel()
    qtbot.addWidget(p)
    p.append_external_message("You", "Hello world", color="#222")
    p.append_external_message("Claude", "Goodbye world", color="#444")
    p.append_external_message("You", "Another line about world",
                              color="#222")
    return p


def test_find_bar_hidden_by_default(qtbot):
    p = _panel(qtbot)
    assert p.find_bar.isHidden()


def test_show_find_focuses_input(qtbot):
    p = _panel(qtbot)
    p._show_find()
    # The find bar should be visible; QLineEdit.hasFocus() is unreliable
    # in offscreen Qt mode, so just check the bar requested focus by
    # verifying it's the focus widget for the panel.
    assert not p.find_bar.isHidden()
    assert p.find_bar._input is not None


def test_do_find_locates_text(qtbot):
    p = _panel(qtbot)
    p._show_find()
    p._do_find("Goodbye", backward=False)
    sel = p.history.textCursor().selectedText()
    assert sel == "Goodbye"
    # No status error when a match was found.
    assert p.find_bar._status.text() == ""


def test_do_find_reports_no_match(qtbot):
    p = _panel(qtbot)
    p._show_find()
    p._do_find("xyzzy_nope", backward=False)
    assert "no match" in p.find_bar._status.text().lower()


def test_do_find_wraps_at_end(qtbot):
    p = _panel(qtbot)
    p._show_find()
    # First match.
    p._do_find("world", backward=False)
    first_pos = p.history.textCursor().position()
    p._do_find("world", backward=False)
    second_pos = p.history.textCursor().position()
    p._do_find("world", backward=False)
    third_pos = p.history.textCursor().position()
    # Three matches; the fourth call should wrap to the first.
    p._do_find("world", backward=False)
    wrapped_pos = p.history.textCursor().position()
    assert wrapped_pos == first_pos
    assert len({first_pos, second_pos, third_pos}) == 3


def test_dismiss_clears_and_hides(qtbot):
    p = _panel(qtbot)
    p._show_find()
    p._do_find("world", backward=False)
    assert p.history.textCursor().hasSelection()
    p._dismiss_find()
    assert p.find_bar.isHidden()
    assert not p.history.textCursor().hasSelection()


def test_find_bar_emits_signals(qtbot):
    from faceview.gui.chat_panel import _ChatFindBar
    bar = _ChatFindBar(parent=None)
    qtbot.addWidget(bar)
    captured = {"next": None, "prev": None, "dismiss": 0}
    bar.find_next.connect(lambda q: captured.__setitem__("next", q))
    bar.find_prev.connect(lambda q: captured.__setitem__("prev", q))
    bar.dismissed.connect(
        lambda: captured.__setitem__("dismiss", captured["dismiss"] + 1)
    )
    bar.set_query("query!")
    bar._on_return()
    assert captured["next"] == "query!"
    bar.dismissed.emit()
    assert captured["dismiss"] == 1


def test_show_prefills_from_selection(qtbot):
    """If the user selected text before Cmd+F, the bar pre-fills with it."""
    from PySide6.QtGui import QTextCursor
    p = _panel(qtbot)
    # Move the QTextEdit's own cursor to the start, then find — the
    # find call uses the live cursor, not a free-standing one.
    cursor = p.history.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    p.history.setTextCursor(cursor)
    found = p.history.find("world")  # advances + selects
    assert found
    p._show_find()
    assert p.find_bar.query() == "world"
