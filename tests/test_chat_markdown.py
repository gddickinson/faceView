"""Coverage for U2 — markdown rendering in the chat panel.

The renderer helpers don't require a full QApplication thanks to
the offscreen Qt platform configured in conftest. The block-list
behaviour is exercised end-to-end through ChatPanel + pytest-qt's
``qtbot``.
"""

from __future__ import annotations


# ── pure renderer helpers ────────────────────────────────────────


def test_render_markdown_inner_html_extracts_body(app):
    from faceview.gui.chat_panel import _render_markdown_to_inner_html
    html = _render_markdown_to_inner_html("**bold** and *italic*")
    # QTextDocument's commonmark renderer emits font-weight CSS rather
    # than <strong>/<em> tags — what matters is the surrounding
    # <body>...</body> got stripped.
    assert "<body" not in html
    # And we didn't return the empty string.
    assert html.strip()


def test_render_markdown_code_fence_becomes_pre(app):
    from faceview.gui.chat_panel import _render_markdown_to_inner_html
    md = "Here:\n\n```python\nprint('hi')\n```\n"
    html = _render_markdown_to_inner_html(md)
    # Qt's renderer wraps code blocks in <pre>; some versions use
    # <pre><code> while others just <pre>. Either is fine.
    assert "<pre" in html
    assert "print" in html


def test_render_block_html_includes_who_header(app):
    from faceview.gui.chat_panel import ChatBlock, _render_block_html
    blk = ChatBlock(who="You", color="#1a73e8", text="hello")
    html = _render_block_html(blk)
    assert "You:" in html
    assert "#1a73e8" in html
    assert "hello" in html


def test_render_block_html_streaming_escapes_lt_gt(app):
    """Streaming view stays plain-text and HTML-escapes user content
    so a partial '<script>' token can't break the panel."""
    from faceview.gui.chat_panel import ChatBlock, _render_block_html
    blk = ChatBlock(who="Claude", color="#000",
                    text="<script>alert(1)</script>", is_streaming=True)
    html = _render_block_html(blk)
    assert "&lt;script&gt;" in html
    assert "<script>alert(1)</script>" not in html


# ── ChatPanel integration ────────────────────────────────────────


def _make_panel(qtbot):
    from faceview.gui.chat_panel import ChatPanel
    panel = ChatPanel()
    qtbot.addWidget(panel)
    return panel


def test_chat_panel_accumulates_blocks(qtbot):
    panel = _make_panel(qtbot)
    panel.append_external_message("Alice", "Hello!", color="#222")
    panel.append_external_message("Bob", "Hi back.", color="#444")
    assert len(panel._blocks) == 2
    assert panel._blocks[0].who == "Alice"
    assert panel._blocks[1].text == "Hi back."


def test_streaming_token_then_reply_renders_markdown(qtbot):
    from faceview.core.events import ChatMessage
    panel = _make_panel(qtbot)
    # Simulate streaming tokens.
    panel._on_token("Here ")
    panel._on_token("is ")
    panel._on_token("**bold**.")
    assert panel._live_block is not None
    assert panel._live_block.is_streaming is True
    # Finalisation.
    panel._on_reply(ChatMessage("assistant", "Here is **bold**."))
    assert panel._live_block is None
    assert len(panel._blocks) == 1
    assert panel._blocks[0].is_streaming is False
    # The QTextEdit's content should now contain font-weight styling
    # rather than the literal `**bold**`.
    html = panel.history.toHtml()
    assert "**bold**" not in html


def test_llm_error_drops_in_flight_streaming(qtbot):
    panel = _make_panel(qtbot)
    panel._on_token("partial ")
    panel._on_token("stuff")
    assert panel._live_block is not None
    panel._on_llm_error("rate limit")
    # Live block discarded; error block in its place.
    assert panel._live_block is None
    assert len(panel._blocks) == 1
    assert panel._blocks[0].who == "error"


def test_seed_demo_uses_markdown(qtbot):
    panel = _make_panel(qtbot)
    panel.seed_demo_conversation()
    assert len(panel._blocks) == 4
    # The fourth block contains a code fence.
    assert "```python" in panel._blocks[3].text
    html = panel.history.toHtml()
    # Rendered output should NOT contain the raw code-fence markers.
    assert "```python" not in html
    assert "<pre" in html
