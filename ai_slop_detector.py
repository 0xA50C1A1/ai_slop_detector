#!/usr/bin/env python3
r"""
AI-slop phrase detector for JSONL conversation datasets.
Reads patterns from patterns.toml and scans messages using the `regex` module.

TOML format:

    [[phrase]]
    category = "romance_nsfw"
    pattern = '(?<!\w)husky\s+voice(?!\w)'
    note = "optional comment, ignored by the script"

Performance:
    Plain literal patterns of the form (?<!\w)escaped_text(?!\w) get merged
    into a single combined regex per category, which is a lot faster than
    running thousands of tiny patterns separately on large datasets (10k+
    lines). Anything with actual regex logic in it is compiled and run on
    its own.
"""
import json
import argparse
import sys
import re as _re_std
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import regex
from tqdm import tqdm

try:
    import tomllib  # stdlib on 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # `pip install tomli` on older Pythons

# ------------------------------------------------------------------
# Populated once per worker process via the pool initializer.
# ------------------------------------------------------------------
# dict[category] -> (compiled_regex, [phrase_labels])
_combined_by_category = None
_complex_rules = None          # list[{"regex": Pattern, "category": str}]
_include_user = None

_WRAP_PREFIX = r"(?<!\w)"
_WRAP_SUFFIX = r"(?!\w)"


def _extract_literal_core(pattern: str):
    r"""
    Check whether `pattern` is just a plain literal wrapped as
    (?<!\w)ESCAPED_TEXT(?!\w), where ESCAPED_TEXT is regex.escape(phrase)
    with escaped spaces turned into \s+.

    Returns the reconstructed human-readable phrase (handy as a label), or
    None if the pattern doesn't fit that shape.
    """
    if not (pattern.startswith(_WRAP_PREFIX) and pattern.endswith(_WRAP_SUFFIX)):
        return None

    core = pattern[len(_WRAP_PREFIX):-len(_WRAP_SUFFIX)]

    # Guard against an empty core (e.g. "(?<!\w)(?!\w)"), which would
    # otherwise be treated as a valid zero-length literal and match at
    # every position in every string once merged into the combined regex.
    if not core:
        return None

    # `core` has to consist purely of regex.escape()-escaped characters and
    # our own \s+ substitution for spaces — no groups, quantifiers, char
    # classes, etc. Anything fancier counts as a "complex" pattern.
    if r"\s+" not in core and " " not in core:
        candidate_phrase = _unescape_regex_literal(core)
        if candidate_phrase is None or candidate_phrase == "":
            return None
        return candidate_phrase if regex.escape(candidate_phrase) == core else None

    # Split on \s+, each chunk must be a clean regex.escape() of one word.
    words = []
    for token in core.split(r"\s+"):
        word = _unescape_regex_literal(token)
        if not word or regex.escape(word) != token:
            return None
        words.append(word)

    if not words:
        return None

    return " ".join(words)


def _unescape_regex_literal(escaped: str):
    """
    Undo regex.escape() on a string, best-effort. Returns None on anything
    that looks off (the caller double-checks by re-escaping anyway).
    """
    try:
        return _re_std.sub(r"\\(.)", r"\1", escaped)
    except Exception:
        return None


def load_patterns(path):
    """Load patterns from TOML and split them into literal vs. complex."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    entries = data.get("phrase", [])
    if not entries:
        sys.exit(f"No [[phrase]] entries found in {path}")

    literal_by_category = defaultdict(list)   # category -> [phrase, ...]
    complex_rules_raw = []                    # [{"pattern":, "category":}]
    errors = []

    for i, entry in enumerate(entries):
        pattern = entry.get("pattern")
        category = entry.get("category", "uncategorized")
        if not pattern:
            errors.append(f"  entry #{i}: missing 'pattern' field")
            continue

        # Compile first so broken regexes get caught early.
        try:
            regex.compile(pattern, regex.IGNORECASE)
        except regex.error as e:
            errors.append(
                f"  entry #{i} ({category}): failed to compile '{pattern}': {e}")
            continue

        literal = _extract_literal_core(pattern)
        if literal is not None:
            literal_by_category[category].append(literal)
        else:
            complex_rules_raw.append(
                {"pattern": pattern, "category": category})

    if errors:
        print(
            f"[!] {len(errors)} problem(s) while loading patterns:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)

    n_literal = sum(len(v) for v in literal_by_category.values())
    n_complex = len(complex_rules_raw)
    print(
        f"Loaded {n_literal} literal patterns (merged into combined regexes) "
        f"and {n_complex} complex patterns (run individually)",
        file=sys.stderr,
    )

    return literal_by_category, complex_rules_raw


def build_combined_by_category(literal_by_category):
    """
    Build one combined regex per category using named groups p0, p1, ...
    (group names only need to be unique within a single category's regex).
    """
    combined = {}
    for category, phrases in literal_by_category.items():
        parts = []
        labels = []
        for idx, phrase in enumerate(phrases):
            escaped = regex.escape(phrase).replace(r"\ ", r"\s+")
            parts.append(rf"(?<!\w)(?P<p{idx}>{escaped})(?!\w)")
            labels.append(phrase)
        compiled = regex.compile("|".join(parts), regex.IGNORECASE)
        combined[category] = (compiled, labels)
    return combined


def compile_complex_rules(complex_rules_raw):
    return [
        {"regex": regex.compile(
            rule["pattern"], regex.IGNORECASE), "category": rule["category"]}
        for rule in complex_rules_raw
    ]


def norm_role(role):
    role = role.lower()
    if role == "gpt":
        return "assistant"
    if role == "human":
        return "user"
    return role


def extract_assistant_text(obj, include_user=False):
    """Pull out assistant (and optionally user) text from a few common dataset formats."""
    texts = []

    def add(role, content):
        role = norm_role(role)
        if role == "assistant" or (include_user and role == "user"):
            texts.append(content or "")

    if isinstance(obj, dict):
        for m in obj.get("messages", []):
            add(m.get("role", ""), m.get("content", ""))
        for c in obj.get("conversations", []):
            add(c.get("from", ""), c.get("value", ""))
        for key in ("prompt", "completion"):
            val = obj.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        add(item.get("role", ""), item.get("content", ""))
            elif isinstance(val, str) and key == "completion":
                texts.append(val)

    return "\n".join(texts)


def count_phrases(text):
    """Return a Counter keyed by (category, normalized_phrase) -> count."""
    counter = Counter()

    # One combined literal regex per category.
    for category, (compiled, labels) in _combined_by_category.items():
        for m in compiled.finditer(text):
            group_name = m.lastgroup
            if group_name is None:
                continue
            idx = int(group_name[1:])  # "p42" -> 42
            counter[(category, labels[idx])] += 1

    # Complex patterns, run one by one.
    for rule in _complex_rules:
        for m in rule["regex"].finditer(text):
            hit = regex.sub(r"\s+", " ", m.group(0).strip()).lower()
            if hit:
                counter[(rule["category"], hit)] += 1

    return counter


def init_worker(combined_by_category, complex_rules, include_user):
    global _combined_by_category, _complex_rules, _include_user
    _combined_by_category = combined_by_category
    _complex_rules = complex_rules
    _include_user = include_user


def process_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    text = extract_assistant_text(obj, _include_user)
    return count_phrases(text) if text else None


def process_chunk(lines):
    chunk_total = Counter()
    for line in lines:
        cnt = process_line(line)
        if cnt:
            chunk_total.update(cnt)
    return chunk_total


def chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _escape_md(text: str) -> str:
    """Escape characters that would break markdown table formatting."""
    return text.replace("\\", "\\\\").replace("|", r"\|").replace("\n", " ").replace("\r", "")


def format_report_md(total: Counter):
    """Build a markdown report: a category summary table + one table per category."""
    by_category = defaultdict(Counter)
    for (category, phrase), count in total.items():
        by_category[category][phrase] = count

    category_totals = {cat: sum(c.values()) for cat, c in by_category.items()}
    total_hits = sum(total.values())

    sorted_categories = sorted(
        category_totals.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )

    lines_out = []

    # --- Summary table -------------------------------------------------
    lines_out.append("## Summary")
    lines_out.append("")
    lines_out.append("| Category | Unique phrases | Total hits | Share |")
    lines_out.append("|---|---:|---:|---:|")
    for category, cat_total in sorted_categories:
        n_unique = len(by_category[category])
        share = f"{(cat_total / total_hits * 100):.1f}%" if total_hits else "0.0%"
        lines_out.append(
            f"| {_escape_md(category)} | {n_unique} | {cat_total} | {share} |"
        )
    lines_out.append(f"| **Total** | "
                      f"**{sum(len(c) for c in by_category.values())}** | "
                      f"**{total_hits}** | **100.0%** |")
    lines_out.append("")

    # --- Per-category detail tables -------------------------------------
    for category, cat_total in sorted_categories:
        lines_out.append(f"## {category} (total: {cat_total})")
        lines_out.append("")
        lines_out.append("| Phrase | Count |")
        lines_out.append("|---|---:|")
        # sort by count desc, then alphabetically for stable output
        for phrase, count in sorted(
            by_category[category].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines_out.append(f"| {_escape_md(phrase)} | {count} |")
        lines_out.append("")

    return lines_out, total_hits


def main():
    p = argparse.ArgumentParser(
        description="AI-slop phrase detector for JSONL datasets")
    p.add_argument("jsonl")
    p.add_argument("-p", "--patterns", default="patterns.toml",
                   help="Path to the TOML patterns file (default: patterns.toml)")
    p.add_argument(
        "-o", "--out", help="Summary report output file instead of stdout")
    p.add_argument("--include-user", action="store_true",
                   help="Also scan user messages")
    p.add_argument("-j", "--jobs", type=int, default=None,
                   help="Number of worker processes (default: all cores)")
    p.add_argument("--chunk-size", type=int, default=200,
                   help="Lines per chunk")
    args = p.parse_args()

    literal_by_category, complex_rules_raw = load_patterns(args.patterns)
    combined_by_category = build_combined_by_category(literal_by_category)
    complex_rules = compile_complex_rules(complex_rules_raw)

    with open(args.jsonl, encoding="utf-8") as f:
        lines = f.readlines()

    total = Counter()
    chunks = list(chunked(lines, args.chunk_size))

    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=init_worker,
        initargs=(combined_by_category, complex_rules, args.include_user),
    ) as executor:
        for chunk_result in tqdm(
            executor.map(process_chunk, chunks),
            total=len(chunks),
            desc="Chunks",
            unit="chunk",
        ):
            total.update(chunk_result)

    lines_out, total_hits = format_report_md(total)
    report = "\n".join(lines_out)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"# AI-slop report\n\n")
            f.write(report)
            f.write(f"\n**Total AI-slop hits: {total_hits}**\n")
    else:
        print("# AI-slop report\n")
        print(report)
        print(f"**Total AI-slop hits: {total_hits}**")


if __name__ == "__main__":
    main()
