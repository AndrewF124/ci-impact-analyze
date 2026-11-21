"""Recognize fixture edits that do not change what the pinned loaders yield."""

SQL_LOADERS = {"load_sql_fixtures": "lines", "load_sql_fixture_pairs": "pairs"}


def fixture_open_key(path, caller_path, caller_name):
    """Tag an observed SQL file open with its reader's input contract."""
    if not path.startswith("tests/fixtures/") or not path.endswith(".sql"):
        return None
    kind = SQL_LOADERS.get(caller_name, "raw") if caller_path == "tests/helpers.py" else "raw"
    return _reader_key(path, kind)


def _reader_key(path, kind):
    return f"fixture-sql:{kind}:{path}"


def _filtered_lines(text):
    # The pinned helpers remove empty lines and column-zero SQL comments.
    return tuple(line for line in text.splitlines() if line and not line.startswith("--"))


def _pair_input(lines):
    # The helper strips each consumed segment before parsing metadata or yielding it.
    parts = "\n".join(lines).split(";")
    return tuple(part.strip() for part in parts[:len(parts) // 2 * 2])


def unaffected_fixture_readers(path, before, after, baseline, isolated):
    """Return tests whose every observed read sees equivalent loader input.

    Require positive attribution in each coverage source that observed a read.
    A raw read, a different loader, or absent provenance prevents exclusion.
    """
    old_lines, new_lines = _filtered_lines(before), _filtered_lines(after)
    equivalent = set()
    if old_lines == new_lines:
        equivalent.update(SQL_LOADERS.values())
    elif _pair_input(old_lines) == _pair_input(new_lines):
        equivalent.add("pairs")
    if not equivalent:
        return 0

    def observations(read_lines):
        return {kind: read_lines(_reader_key(path, kind)).get(0, 0)
                for kind in (*SQL_LOADERS.values(), "raw")}

    proved = unproved = 0
    for read_lines in (baseline, lambda path: isolated.get(path, {})):
        readers = read_lines(path).get(0, 0)
        if not readers:
            continue
        allowed = blocked = 0
        for kind, mask in observations(read_lines).items():
            if kind in equivalent:
                allowed |= mask
            else:
                blocked |= mask
        proved |= readers & allowed
        unproved |= readers & (~allowed | blocked)
    return proved & ~unproved
