"""
Math formatter utility: converts LaTeX-style math inside explicit delimiters
to readable ASCII/Unicode while leaving non-math text unchanged.

Supported math delimiters (scoped conversion only):
- Inline: \( ... \), $ ... $
- Block: \[ ... \], $$ ... $$

Usage patterns:
- For streaming: create one instance, call feed(chunk) repeatedly, then flush() at end
- For full text: use format_full(text) to process in one call
"""

from typing import Tuple
import re


class MathFormatter:
    """Stateful formatter for streamed text with delimiter-scoped conversion."""

    def __init__(self, use_block_markers: bool = True):
        # Rolling buffer for streamed input
        self._buffer: str = ""
        self._use_block_markers = use_block_markers

        # Precompile regex patterns used in conversions
        self._re_frac = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")
        self._re_sqrt = re.compile(r"\\sqrt\{([^}]+)\}")
        self._re_nsqr = re.compile(r"\\sqrt\[([^\]]+)\]\{([^}]+)\}")
        self._re_sum_limits = re.compile(r"\\sum_\{([^}]+)\}\^\{([^}]+)\}")
        self._re_int_limits = re.compile(r"\\int_\{([^}]+)\}\^\{([^}]+)\}")
        self._re_prod_limits = re.compile(r"\\prod_\{([^}]+)\}\^\{([^}]+)\}")
        self._re_lim_sub = re.compile(r"\\lim_\{([^}]+)\}")
        self._re_sub_braced = re.compile(r"_\{([^}]+)\}")
        self._re_sub_simple = re.compile(r"_([a-zA-Z0-9])")
        self._re_sup_braced = re.compile(r"\^\{([^}]+)\}")
        self._re_sup_simple = re.compile(r"\^([a-zA-Z0-9\+\-])")
        self._re_commands = re.compile(r"\\[a-zA-Z]+")

        # Maps for sub/superscripts and symbols
        self._sub_map = {
            '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆',
            '7': '₇', '8': '₈', '9': '₉', 'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ',
            'j': 'ⱼ', 'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ', 'p': 'ₚ',
            'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ', 'v': 'ᵥ', 'x': 'ₓ', '+': '₊',
            '-': '₋', '=': '₌', '(': '₍', ')': '₎', ' ': ' '
        }
        self._sup_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶',
            '7': '⁷', '8': '⁸', '9': '⁹', 'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ',
            'e': 'ᵉ', 'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ', 'k': 'ᵏ',
            'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ', 'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ',
            't': 'ᵗ', 'u': 'ᵘ', 'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
            '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾', ' ': ' '
        }

        self._greek = {
            r"\alpha": 'α', r"\beta": 'β', r"\gamma": 'γ', r"\delta": 'δ',
            r"\epsilon": 'ε', r"\varepsilon": 'ε', r"\zeta": 'ζ', r"\eta": 'η',
            r"\theta": 'θ', r"\vartheta": 'ϑ', r"\iota": 'ι', r"\kappa": 'κ',
            r"\lambda": 'λ', r"\mu": 'μ', r"\nu": 'ν', r"\xi": 'ξ',
            r"\omicron": 'ο', r"\pi": 'π', r"\varpi": 'ϖ', r"\rho": 'ρ',
            r"\varrho": 'ϱ', r"\sigma": 'σ', r"\varsigma": 'ς', r"\tau": 'τ',
            r"\upsilon": 'υ', r"\phi": 'φ', r"\varphi": 'φ', r"\chi": 'χ',
            r"\psi": 'ψ', r"\omega": 'ω',
            r"\Gamma": 'Γ', r"\Delta": 'Δ', r"\Theta": 'Θ', r"\Lambda": 'Λ',
            r"\Xi": 'Ξ', r"\Pi": 'Π', r"\Sigma": 'Σ', r"\Upsilon": 'Υ',
            r"\Phi": 'Φ', r"\Psi": 'Ψ', r"\Omega": 'Ω'
        }

        self._symbols = {
            # Operators
            r"\sum": 'Σ', r"\prod": '∏', r"\int": '∫', r"\oint": '∮',
            r"\iint": '∬', r"\iiint": '∭', r"\bigcup": '⋃', r"\bigcap": '⋂',
            r"\coprod": '∐', r"\bigoplus": '⊕', r"\bigotimes": '⊗',
            r"\bigwedge": '⋀', r"\bigvee": '⋁',
            # Relations
            r"\leq": '≤', r"\geq": '≥', r"\neq": '≠', r"\approx": '≈',
            r"\equiv": '≡', r"\sim": '∼', r"\simeq": '≃', r"\cong": '≅',
            r"\propto": '∝', r"\perp": '⊥', r"\parallel": '∥',
            r"\subset": '⊂', r"\supset": '⊃', r"\subseteq": '⊆', r"\supseteq": '⊇',
            r"\in": '∈', r"\notin": '∉', r"\ni": '∋',
            # Arrows
            r"\rightarrow": '→', r"\leftarrow": '←', r"\leftrightarrow": '↔',
            r"\Rightarrow": '⇒', r"\Leftarrow": '⇐', r"\Leftrightarrow": '⇔',
            r"\uparrow": '↑', r"\downarrow": '↓', r"\updownarrow": '↕',
            r"\to": '→', r"\mapsto": '↦',
            # Logic
            r"\forall": '∀', r"\exists": '∃', r"\nexists": '∄', r"\land": '∧',
            r"\lor": '∨', r"\lnot": '¬', r"\neg": '¬', r"\because": '∵', r"\therefore": '∴',
            # Sets
            r"\emptyset": '∅', r"\varnothing": '∅', r"\cap": '∩', r"\cup": '∪',
            r"\setminus": '∖', r"\complement": 'ᶜ',
            # Misc
            r"\infty": '∞', r"\partial": '∂', r"\nabla": '∇', r"\pm": '±', r"\mp": '∓',
            r"\times": '×', r"\div": '÷', r"\cdot": '·', r"\bullet": '•', r"\star": '⋆',
            r"\ast": '∗', r"\circ": '∘', r"\oplus": '⊕', r"\ominus": '⊖', r"\otimes": '⊗',
            r"\oslash": '⊘', r"\odot": '⊙', r"\dagger": '†', r"\ddagger": '‡', r"\angle": '∠',
            r"\degree": '°', r"\prime": '′', r"\|": '‖'
        }

        self._functions = [
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'sinh', 'cosh', 'tanh',
            'arcsin', 'arccos', 'arctan', 'log', 'ln', 'exp', 'det', 'dim', 'ker',
            'gcd', 'lcm', 'min', 'max', 'sup', 'inf', 'lim', 'limsup', 'liminf'
        ]

    # Public API -------------------------------------------------------------

    def feed(self, text: str) -> str:
        """Feed streamed text; returns formatted output for any completed parts."""
        if not text:
            return ""
        self._buffer += text
        output_parts = []

        while True:
            open_pos, token = self._find_next_open(self._buffer)
            if open_pos == -1:
                # No more opens; emit everything before any partial open
                break

            # Emit text before open token as-is (non-math)
            if open_pos > 0:
                output_parts.append(self._buffer[:open_pos])
                self._buffer = self._buffer[open_pos:]
                open_pos = 0

            # Attempt to find closing token
            close_pos = self._find_close(self._buffer, token)
            if close_pos == -1:
                # Wait for more text
                break

            # Extract inside content (exclude delimiters)
            content_start, content_end = self._content_slice(token)
            inner = self._buffer[content_start:close_pos]
            converted = self._convert_math(inner)

            # Wrap for block tokens if enabled
            if self._use_block_markers and token in ('\\[', '$$'):
                wrapped = f"\n[EQUATION]\n{converted}\n[/EQUATION]\n"
            else:
                wrapped = converted

            output_parts.append(wrapped)

            # Remove processed span from buffer
            after_close = close_pos + len(self._matching_close(token))
            self._buffer = self._buffer[after_close:]

        # Output any non-math leading text up to last complete open (if any)
        # but keep remaining buffer for potential future math spans
        # Find last open token position; if none, emit everything and clear buffer
        last_open_pos, _ = self._find_next_open(self._buffer)
        if last_open_pos == -1:
            output_parts.append(self._buffer)
            self._buffer = ""

        return ''.join(output_parts)

    def flush(self) -> str:
        """Flush any remaining buffered text (without forcing partial math conversion)."""
        remaining = self._buffer
        self._buffer = ""
        return remaining

    def format_full(self, text: str) -> str:
        """Format a full, non-streamed text in one pass."""
        # Use a fresh instance-like state to avoid mutating streamer buffer
        saved = self._buffer
        self._buffer = ""
        out = self.feed(text) + self.flush()
        self._buffer = saved
        return out

    # Internal helpers -------------------------------------------------------

    @staticmethod
    def _content_slice(token: str) -> Tuple[int, int]:
        if token in ('\\[', '\\('):
            return (2, -2)  # exclude opening \\[ or \\(
        if token in ('$$', '$'):
            return (len(token), -len(token))
        return (0, 0)

    @staticmethod
    def _matching_close(token: str) -> str:
        if token == '\\[':
            return '\\]'
        if token == '\\(':
            return '\\)'
        if token in ('$$', '$'):
            return token
        return ''

    @staticmethod
    def _find_next_open(text: str) -> Tuple[int, str]:
        candidates = []
        for t in ('\\[', '\\(', '$$', '$'):
            pos = text.find(t)
            if pos != -1:
                candidates.append((pos, t))
        if not candidates:
            return -1, ''
        return min(candidates, key=lambda x: x[0])

    def _find_close(self, text: str, token: str) -> int:
        close = self._matching_close(token)
        if not close:
            return -1
        start = len(token)
        return text.find(close, start)

    def _convert_math(self, s: str) -> str:
        # Subscripts
        def _sub_braced(m: re.Match) -> str:
            return ''.join(self._sub_map.get(ch.lower(), ch if ch != ',' else '，') for ch in m.group(1))

        s = self._re_sub_braced.sub(_sub_braced, s)
        s = self._re_sub_simple.sub(lambda m: self._sub_map.get(m.group(1).lower(), '_' + m.group(1)), s)

        # Superscripts (include degree handling ^\circ)
        s = s.replace('^\\circ', '°')
        def _sup_braced(m: re.Match) -> str:
            return ''.join(self._sup_map.get(ch.lower(), ch) for ch in m.group(1))
        s = self._re_sup_braced.sub(_sup_braced, s)
        s = self._re_sup_simple.sub(lambda m: self._sup_map.get(m.group(1).lower(), '^' + m.group(1)), s)

        # Fractions and roots
        s = self._re_frac.sub(r"(\1)/(\2)", s)
        s = self._re_nsqr.sub(r"ⁿ√(\2) where n=\1", s)
        s = self._re_sqrt.sub(r"√(\1)", s)

        # Limits on sum/product/integral and lim
        s = self._re_sum_limits.sub(r"Σ[\1 → \2]", s)
        s = self._re_prod_limits.sub(r"∏[\1 → \2]", s)
        s = self._re_int_limits.sub(r"∫[\1 → \2]", s)
        s = self._re_lim_sub.sub(r"lim[\1]", s)

        # Functions
        for f in self._functions:
            s = s.replace(f"\\{f}", f)

        # Greek letters and symbols
        for k, v in {**self._greek, **self._symbols}.items():
            s = s.replace(k, v)

        # Matrices/arrays within math only
        s = re.sub(r"\\begin\{(?:[bBpPvV]?matrix)\}", '[', s)
        s = re.sub(r"\\end\{(?:[bBpPvV]?matrix)\}", ']', s)
        s = s.replace(' & ', ' ')
        s = s.replace('\\\\', ' | ')

        # Cleanups: remove sizing and spacing commands
        for cmd in ('\\left', '\\right', '\\big', '\\Big', '\\bigg', '\\Bigg'):
            s = s.replace(cmd, '')
        s = s.replace('\\,', ' ').replace('\\ ', ' ').replace('\\;', ' ').replace('\\:', ' ').replace('\\!', '')

        # Remove any remaining commands like \foo
        s = self._re_commands.sub('', s)

        # Strip leftover braces that are LaTeX-specific
        s = s.replace('{', '').replace('}', '')

        return s


