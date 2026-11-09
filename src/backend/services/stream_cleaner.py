"""Applies clean_output's rules to an answer that arrives in pieces.

clean_output needs the whole answer: it drops sentences that claim the products cover the
whole catalog and strips a leading <think> block. When streaming, text is held back only
until a sentence or line is complete, then released through the same rules. So the
shopper sees whole sentences (and whole table rows) as they finish, never a claim that
is retracted later.
"""
import re

from src.backend.services.llm import CATALOG_CLAIM

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


class StreamCleaner:
    def __init__(self) -> None:
        self._buf = ""
        self._raw: list[str] = []
        self._out: list[str] = []
        self._line_has_text = False   # this line already has released (kept) text
        self._line_is_table = None    # decided at the line's first non-space character
        self._think = "undecided"     # undecided | inside | done

    # ── public ────────────────────────────────────────────
    def feed(self, delta: str) -> str:
        """Add model output; returns the text that is now safe to show (may be "")."""
        self._raw.append(delta)
        self._buf += delta
        if not self._skip_think():
            return ""
        return self._release(final=False)

    def finish(self) -> str:
        """Flush what's left at the end of the stream."""
        if self._think != "done":  # an unterminated or undecided think block: keep the text
            self._think = "done"
        released = self._release(final=True)
        if not "".join(self._out).strip() and "".join(self._raw).strip():
            # Everything was dropped as a catalog claim: show it rather than nothing,
            # as clean_output does.
            fallback = "".join(self._raw).strip()
            self._out = [fallback]
            return fallback
        return released

    @property
    def text(self) -> str:
        """The whole cleaned answer so far, as stored in the cache."""
        return re.sub(r"\n{3,}", "\n\n", "".join(self._out)).strip()

    # ── internals ─────────────────────────────────────────
    def _skip_think(self) -> bool:
        """False while text is still inside (or might start) a leading <think> block."""
        if self._think == "done":
            return True
        stripped = self._buf.lstrip()
        if self._think == "undecided":
            if _THINK_OPEN.startswith(stripped) and len(stripped) < len(_THINK_OPEN):
                return False  # could still become "<think>"
            if not stripped.startswith(_THINK_OPEN):
                self._think = "done"
                return True
            self._think = "inside"
        end = self._buf.find(_THINK_CLOSE)
        if end == -1:
            return False
        self._buf = self._buf[end + len(_THINK_CLOSE):].lstrip()
        self._think = "done"
        return True

    def _release(self, final: bool) -> str:
        emitted = []
        while True:
            newline = self._buf.find("\n")
            if newline != -1:
                piece, self._buf = self._buf[:newline], self._buf[newline + 1:]
                emitted.append(self._finish_line(piece))
                continue
            if self._decide_table(self._buf) is False:  # prose: release finished sentences
                ends = [m.end() for m in _SENTENCE_END.finditer(self._buf)]
                if ends:  # everything up to the last finished sentence
                    done, self._buf = self._buf[:ends[-1]], self._buf[ends[-1]:]
                    emitted.append(self._keep_sentences(done))
            if final and self._buf:
                piece, self._buf = self._buf, ""
                emitted.append(self._finish_line(piece, newline=False))
            break
        text = "".join(emitted)
        self._out.append(text)
        return text

    def _decide_table(self, text: str) -> bool | None:
        if self._line_is_table is None and text.strip():
            self._line_is_table = text.lstrip().startswith("|")
        return self._line_is_table

    def _keep_sentences(self, text: str) -> str:
        """Drop whole sentences that claim catalog coverage; keep the rest as written."""
        parts = re.split(r"((?<=[.!?])\s+)", text)  # keep the separators
        kept = []
        for i in range(0, len(parts), 2):
            sentence, sep = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
            if sentence.strip() and CATALOG_CLAIM.search(sentence):
                continue
            kept.append(sentence + sep)
        out = "".join(kept)
        if out.strip():
            self._line_has_text = True
        return out

    def _finish_line(self, piece: str, newline: bool = True) -> str:
        is_table = self._decide_table(piece)
        had_text_before = self._line_has_text
        out = piece if is_table else self._keep_sentences(piece)
        line_dropped = not out.strip() and not had_text_before and piece.strip()
        self._line_has_text, self._line_is_table = False, None
        if line_dropped:
            return ""  # the whole line was a catalog claim: drop its newline too
        return out + ("\n" if newline else "")
