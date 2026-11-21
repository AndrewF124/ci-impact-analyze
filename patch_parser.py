"""Parse textual Git patches and apply their hunks against pinned source."""

from __future__ import annotations

import re
from dataclasses import dataclass


class InvalidPatch(ValueError):
    """The patch is malformed or uses a format the selector cannot verify."""


@dataclass(frozen=True)
class DiffLine:
    kind: str
    text: str
    no_newline: bool

    def source_line(self) -> str:
        return self.text if self.no_newline else self.text + "\n"


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[DiffLine, ...]

    @property
    def old_offset(self) -> int:
        return self.old_start - bool(self.old_count)

    @property
    def new_offset(self) -> int:
        return self.new_start - bool(self.new_count)


@dataclass(frozen=True)
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: tuple[Hunk, ...]

    @property
    def changed_old_lines(self) -> frozenset[int]:
        return self._changed_lines("-")

    @property
    def changed_new_lines(self) -> frozenset[int]:
        return self._changed_lines("+")

    def _changed_lines(self, kind: str) -> frozenset[int]:
        changed = set()
        for hunk in self.hunks:
            number = (hunk.old_offset if kind == "-" else hunk.new_offset) + 1
            for line in hunk.lines:
                if line.kind == kind:
                    changed.add(number)
                if line.kind in (" ", kind):
                    number += 1
        return frozenset(changed)

    def apply(self, original: str) -> str:
        """Apply a textual patch after verifying all old lines and context."""
        if self.old_path is None and original:
            raise InvalidPatch("an added file must have empty original text")

        pieces = original.split("\n")
        old_lines = [piece + "\n" for piece in pieces[:-1]]
        if pieces[-1]:
            old_lines.append(pieces[-1])

        output = []
        offset = 0
        for hunk in self.hunks:
            if hunk.old_offset + hunk.old_count > len(old_lines):
                raise InvalidPatch("hunk extends beyond the end of the original file")
            output.extend(old_lines[offset : hunk.old_offset])
            offset = hunk.old_offset
            for line in hunk.lines:
                source = line.source_line()
                if line.kind != "+":
                    if old_lines[offset] != source:
                        raise InvalidPatch("hunk does not match pinned source")
                    offset += 1
                if line.kind != "-":
                    output.append(source)

        output.extend(old_lines[offset:])
        if any(not line.endswith("\n") for line in output[:-1]):
            raise InvalidPatch("a missing newline marker must end the file")
        if self.new_path is None and output:
            raise InvalidPatch("a deleted file must have empty resulting text")
        return "".join(output)


_DIFF_HEADER = "diff --git "
_HUNK_HEADER = re.compile(
    r"@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@(?: .*)?"
)
_INDEX_LINE = re.compile(r"index [a-f0-9]+\.\.[a-f0-9]+(?: (?:100644|100755))?")
_FILE_MODE = re.compile(r"(new|deleted) file mode (?:100644|100755)")
_NO_NEWLINE = r"\ No newline at end of file"
_PATH_ESCAPES = {
    "a": b"\a",
    "b": b"\b",
    "t": b"\t",
    "n": b"\n",
    "v": b"\v",
    "f": b"\f",
    "r": b"\r",
    '"': b'"',
    "\\": b"\\",
}


def _quoted_path(value: str) -> tuple[str, str]:
    result = bytearray()
    position = 1
    while position < len(value):
        character = value[position]
        position += 1
        if character == '"':
            try:
                return result.decode("utf-8"), value[position:]
            except UnicodeDecodeError as error:
                raise InvalidPatch("a quoted path must decode as UTF-8") from error
        if character != "\\":
            result.extend(character.encode("utf-8"))
            continue
        escape = value[position : position + 1]
        if escape in _PATH_ESCAPES:
            result.extend(_PATH_ESCAPES[escape])
            position += 1
            continue
        octal = value[position : position + 3]
        if re.fullmatch(r"[0-3][0-7]{2}", octal) is None:
            raise InvalidPatch("unsupported escape in a quoted path")
        result.append(int(octal, 8))
        position += 3
    raise InvalidPatch("unterminated quoted path")


def _file_path(value: str, prefix: str) -> str | None:
    if value.startswith('"'):
        value, suffix = _quoted_path(value)
    else:
        value, separator, rest = value.partition("\t")
        suffix = separator + rest
    if suffix not in ("", "\t"):
        raise InvalidPatch("unexpected data after a file path")
    if value == "/dev/null":
        return None
    if not value.startswith(prefix):
        raise InvalidPatch("missing Git path prefix")
    path = value[len(prefix) :]
    if any(part in ("", ".", "..") for part in path.split("/")) or any(
        character in path for character in ("\0", "\\", ":")
    ):
        raise InvalidPatch("unsafe repository path")
    return path


def _check_diff_header(header: str, old_path: str | None, new_path: str | None) -> None:
    # Git leaves spaces in unquoted paths. Match the known paths from the
    # old/new markers instead of trying to split the header on whitespace.
    old_name = "a/" + (old_path or new_path or "")
    new_name = "b/" + (new_path or old_path or "")
    if header.startswith('"'):
        source, remainder = _quoted_path(header)
        if source != old_name:
            raise InvalidPatch("inconsistent old file path")
    else:
        if not header.startswith(old_name + " "):
            raise InvalidPatch("inconsistent old file path")
        remainder = header[len(old_name) :]
    if not remainder.startswith(" "):
        raise InvalidPatch("missing new path in diff header")
    target = remainder[1:]
    if target.startswith('"'):
        target, suffix = _quoted_path(target)
        if suffix:
            raise InvalidPatch("unexpected data after the diff header")
    if target != new_name:
        raise InvalidPatch("inconsistent new file path")


def _read_hunk(lines: list[str], index: int) -> tuple[Hunk, int]:
    match = _HUNK_HEADER.fullmatch(lines[index])
    if match is None:
        raise InvalidPatch("invalid hunk header")
    try:
        old_start, old_count, new_start, new_count = (int(value or 1) for value in match.groups())
    except ValueError as error:
        raise InvalidPatch("invalid hunk line numbers") from error
    if (old_count and old_start == 0) or (new_count and new_start == 0):
        raise InvalidPatch("invalid hunk line numbers")

    index += 1
    old_remaining, new_remaining = old_count, new_count
    entries = []
    while old_remaining or new_remaining:
        if index == len(lines):
            raise InvalidPatch("incomplete hunk")
        row = lines[index]
        if not row or row[0] not in " +-":
            raise InvalidPatch("unexpected hunk line")
        kind, source = row[0], row[1:]
        old_remaining -= kind != "+"
        new_remaining -= kind != "-"
        if old_remaining < 0 or new_remaining < 0:
            raise InvalidPatch("hunk lines exceed the stated lengths")
        index += 1
        no_newline = index < len(lines) and lines[index] == _NO_NEWLINE
        if no_newline:
            if not source or (kind != "+" and old_remaining) or (kind != "-" and new_remaining):
                raise InvalidPatch("misplaced missing newline marker")
            index += 1
        entries.append(DiffLine(kind, source, no_newline))
    if not any(line.kind != " " for line in entries):
        raise InvalidPatch("hunk has no changed lines")
    return Hunk(old_start, old_count, new_start, new_count, tuple(entries)), index


def parse_patch(text: str) -> list[FilePatch]:
    """Read Git unified diffs, rejecting unsupported file operations."""
    if not text:
        raise InvalidPatch("empty patch")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if len(lines) > 1 and lines[0].endswith("\r"):
        # Handle CRLF transport without discarding CRs inside original lines.
        lines = [line.removesuffix("\r") for line in lines]

    files = []
    paths = set()
    index = 0
    while index < len(lines):
        header = lines[index]
        if not header.startswith(_DIFF_HEADER):
            raise InvalidPatch("expected a Git diff header")
        index += 1
        mode = None
        has_index = False
        while index < len(lines) and not lines[index].startswith("--- "):
            metadata = lines[index]
            if not has_index and _INDEX_LINE.fullmatch(metadata):
                has_index = True
            elif mode is None and (match := _FILE_MODE.fullmatch(metadata)):
                mode = match[1]
            else:
                raise InvalidPatch("unsupported file metadata")
            index += 1
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise InvalidPatch("missing old or new file path")
        old_path = _file_path(lines[index][4:], "a/")
        new_path = _file_path(lines[index + 1][4:], "b/")
        if old_path is None and new_path is None:
            raise InvalidPatch("patch must name a source or target file")
        if old_path and new_path and old_path != new_path:
            raise InvalidPatch("file renames are unsupported")
        if (mode == "new" and old_path is not None) or (
            mode == "deleted" and new_path is not None
        ):
            raise InvalidPatch("file mode conflicts with old or new path")
        _check_diff_header(header[len(_DIFF_HEADER) :], old_path, new_path)
        path = old_path or new_path
        if path in paths:
            raise InvalidPatch("file appears in more than one diff")
        paths.add(path)
        index += 2

        hunks = []
        previous_old_end = previous_new_end = 0
        old_no_newline = new_no_newline = False
        while index < len(lines) and lines[index].startswith("@@"):
            hunk, index = _read_hunk(lines, index)
            old_gap = hunk.old_offset - previous_old_end
            new_gap = hunk.new_offset - previous_new_end
            if old_gap < 0 or old_gap != new_gap or old_no_newline or new_no_newline:
                raise InvalidPatch("inconsistent hunk locations")
            if (old_path is None and (hunk.old_offset or hunk.old_count)) or (
                new_path is None and (hunk.new_offset or hunk.new_count)
            ):
                raise InvalidPatch("hunk names lines in a nonexistent file")
            previous_old_end = hunk.old_offset + hunk.old_count
            previous_new_end = hunk.new_offset + hunk.new_count
            old_no_newline = any(line.no_newline and line.kind != "+" for line in hunk.lines)
            new_no_newline = any(line.no_newline and line.kind != "-" for line in hunk.lines)
            hunks.append(hunk)
        if not hunks:
            raise InvalidPatch("textual file patch has no hunks")
        files.append(FilePatch(old_path, new_path, tuple(hunks)))
    return files
