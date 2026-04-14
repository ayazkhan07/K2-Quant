"""
Lightweight markdown-to-HTML converter for Thinkspace chat bubbles.

Handles the subset of markdown that LLMs typically produce:
  - Bold, italic, bold+italic
  - Headers (##, ###, ####)
  - Bullet lists (-, *, nested via indentation)
  - Numbered lists (1., 2., nested via indentation)
  - Paragraph breaks (double newline)
  - Inline code (`code`)
  - Indentation preservation

Does NOT handle: images, links, horizontal rules, blockquotes.
Pipe tables and fenced code blocks are handled separately in right_pane.py.
"""

import re
import html as html_mod
from typing import List


def markdown_to_html(text: str) -> str:
    """Convert markdown text to styled HTML for QLabel rich text."""
    lines = text.split('\n')
    out: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # blank line -> paragraph break
        if not stripped:
            out.append('<br>')
            i += 1
            continue

        # header (## / ### / ####)
        hdr_match = re.match(r'^(#{1,4})\s+(.+)$', stripped)
        if hdr_match:
            level = len(hdr_match.group(1))
            content = _inline_format(hdr_match.group(2))
            sizes = {1: 16, 2: 14, 3: 13, 4: 12}
            sz = sizes.get(level, 12)
            out.append(
                f'<div style="margin:8px 0 4px 0; font-size:{sz}px; '
                f'font-weight:600; color:#e0e0e0;">{content}</div>')
            i += 1
            continue

        # bullet list block
        if _is_bullet(stripped):
            block, i = _collect_list_block(lines, i, _is_bullet)
            out.append(_render_bullet_list(block))
            continue

        # numbered list block
        if _is_numbered(stripped):
            block, i = _collect_list_block(lines, i, _is_numbered_or_continuation)
            out.append(_render_numbered_list(block))
            continue

        # regular text line
        content = _inline_format(stripped)
        out.append(f'<span style="color:#ffffff;">{content}</span><br>')
        i += 1

    return ''.join(out)


# -- inline formatting ----------------------------------------------------

def _inline_format(text: str) -> str:
    """Apply inline markdown: bold, italic, inline code."""
    text = html_mod.escape(text)

    # inline code `...`
    text = re.sub(
        r'`([^`]+)`',
        r'<span style="background:#1a1a1a; color:#a8c7e8; '
        r'padding:1px 4px; border-radius:2px; '
        r'font-family:Consolas,monospace; font-size:11px;">\1</span>',
        text)

    # bold + italic ***text*** or ___text___
    text = re.sub(
        r'\*\*\*(.+?)\*\*\*',
        r'<b><i>\1</i></b>', text)

    # bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # italic *text* or _text_ (but not mid-word underscores)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<i>\1</i>', text)

    return text


# -- list detection --------------------------------------------------------

def _is_bullet(line: str) -> bool:
    stripped = line.lstrip()
    return bool(re.match(r'^[-*+]\s', stripped))


def _is_numbered(line: str) -> bool:
    stripped = line.lstrip()
    return bool(re.match(r'^\d+[\.\)]\s', stripped))


def _is_numbered_or_continuation(line: str) -> bool:
    return _is_numbered(line.strip()) or _is_bullet(line.strip())


def _get_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _collect_list_block(lines: List[str], start: int,
                        predicate) -> tuple:
    """Collect consecutive list lines (including nested) until a non-list line."""
    block = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            # blank line inside list -- peek ahead
            if i + 1 < len(lines) and predicate(lines[i + 1]):
                block.append(lines[i])
                i += 1
                continue
            break
        if predicate(lines[i]) or _get_indent(lines[i]) > 0:
            block.append(lines[i])
            i += 1
        else:
            break
    return block, i


# -- list rendering --------------------------------------------------------

def _render_bullet_list(block: List[str]) -> str:
    """Render a block of bullet-list lines into nested <ul>/<li> HTML."""
    items = _parse_list_items(block, _bullet_content)
    return _items_to_html(items, ordered=False)


def _render_numbered_list(block: List[str]) -> str:
    """Render a block of numbered-list lines into nested <ol>/<li> HTML."""
    items = _parse_list_items(block, _numbered_content)
    return _items_to_html(items, ordered=True)


def _bullet_content(line: str) -> str:
    stripped = line.lstrip()
    m = re.match(r'^[-*+]\s+(.*)$', stripped)
    return m.group(1) if m else stripped


def _numbered_content(line: str) -> str:
    stripped = line.lstrip()
    m = re.match(r'^\d+[\.\)]\s+(.*)$', stripped)
    return m.group(1) if m else stripped


def _parse_list_items(block: List[str], content_fn) -> list:
    """Parse list lines into [(indent_level, content_html), ...]."""
    if not block:
        return []

    items = []
    base_indent = _get_indent(block[0])

    for line in block:
        if not line.strip():
            continue
        indent = _get_indent(line)
        level = max(0, (indent - base_indent) // 2)
        content = content_fn(line)
        items.append((level, _inline_format(content)))

    return items


def _items_to_html(items: list, ordered: bool) -> str:
    """Convert [(level, html), ...] into nested list HTML."""
    if not items:
        return ""

    tag = "ol" if ordered else "ul"
    style = (
        'style="margin:4px 0 4px 0; padding-left:20px; color:#ffffff; '
        'line-height:1.6;"'
    )

    html_parts = [f'<{tag} {style}>']
    current_level = 0

    for level, content in items:
        while level > current_level:
            html_parts.append(f'<{tag} style="padding-left:18px; margin:2px 0;">')
            current_level += 1
        while level < current_level:
            html_parts.append(f'</{tag}>')
            current_level -= 1
        html_parts.append(f'<li style="margin:2px 0;">{content}</li>')

    while current_level > 0:
        html_parts.append(f'</{tag}>')
        current_level -= 1

    html_parts.append(f'</{tag}>')
    return ''.join(html_parts)
