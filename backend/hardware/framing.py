"""
Line framer — splits a stream of bytes into discrete lines.

The PHASE 14 wire protocol is one ASCII line per frame, terminated
by ``\n`` (LF). CR/LF (``\r\n``) is also accepted (the CR is
stripped). The framer holds an internal buffer; ``feed()`` returns
zero or more complete lines on each call.

If a line exceeds ``max_line_bytes`` without a newline, it is
flushed to the overflow counter and the buffer is reset. The
decoder counts these as ``format_errors``.
"""

from __future__ import annotations

from typing import List


class LineFramer:
    """Byte stream → list of complete lines."""

    def __init__(self, max_line_bytes: int = 8192) -> None:
        if max_line_bytes <= 0:
            raise ValueError("max_line_bytes must be > 0")
        self._max = int(max_line_bytes)
        self._buf = bytearray()
        self._overflow_lines: int = 0

    # ------------------------------------------------------------------
    @property
    def buffered_bytes(self) -> int:
        return len(self._buf)

    @property
    def overflow_lines(self) -> int:
        """Number of lines that exceeded ``max_line_bytes`` and were dropped."""
        return self._overflow_lines

    # ------------------------------------------------------------------
    def feed(self, chunk: bytes) -> List[bytes]:
        """Append ``chunk`` to the buffer and return any complete lines."""
        if not chunk:
            return []
        lines: List[bytes] = []
        # Coerce to bytes; allow bytearray/memoryview callers to pass their slice.
        data = bytes(chunk)
        self._buf.extend(data)
        while True:
            idx = self._buf.find(b"\n")
            if idx < 0:
                # No more complete lines.
                if len(self._buf) > self._max:
                    # Discard the runaway buffer; count as one overflow.
                    self._overflow_lines += 1
                    self._buf.clear()
                return lines
            # Reject an oversize in-progress line: count as one
            # overflow, drop the bytes including the newline, and
            # continue scanning the rest of the buffer.
            if idx > self._max:
                del self._buf[: idx + 1]
                self._overflow_lines += 1
                continue
            line = bytes(self._buf[:idx])
            del self._buf[: idx + 1]
            # Strip a trailing CR (CRLF) without penalising mid-line CRs.
            if line.endswith(b"\r"):
                line = line[:-1]
            lines.append(line)

    def reset(self) -> None:
        self._buf.clear()
        # Overflow counter is intentionally preserved across resets —
        # it is a running diagnostic, not session state.

    def flush_incomplete(self) -> bytes:
        """Return and clear any partial trailing line.

        Useful at shutdown to surface the last partial frame as a
        diagnostic rather than dropping it silently. The partial
        line is NOT counted as an overflow.
        """
        if not self._buf:
            return b""
        out = bytes(self._buf)
        self._buf.clear()
        return out


__all__ = ["LineFramer"]
