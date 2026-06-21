#!/usr/bin/env python3

import argparse
import fnmatch
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Add your specific long path here.
# Using a set of strings for quick lookup.
EXCLUDED_PATHS = {"node_modules", ".git", "main.py"}


@dataclass
class GitIgnoreRule:
    pattern: str
    negated: bool
    directory_only: bool
    anchored: bool


def load_gitignore(base_path: Path) -> list[GitIgnoreRule]:
    gitignore = base_path / ".gitignore"
    rules: list[GitIgnoreRule] = []
    if not gitignore.exists():
        return rules

    with gitignore.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:].strip()
            directory_only = line.endswith("/")
            if directory_only:
                line = line.rstrip("/")
            anchored = line.startswith("/")
            if anchored:
                line = line.lstrip("/")
            if not line:
                continue
            rules.append(GitIgnoreRule(line, negated, directory_only, anchored))
    return rules


def _match_rule(rel_path: str, is_dir: bool, rule: GitIgnoreRule) -> bool:
    pattern = rule.pattern
    if rule.directory_only and not is_dir and not rel_path.startswith(pattern + "/"):
        return False
    if "/" not in pattern:
        parts = rel_path.split("/")
        return any(fnmatch.fnmatch(part, pattern) for part in parts)
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    if rule.directory_only and rel_path.startswith(pattern + "/"):
        return True
    return False


def is_ignored(path: Path, base: Path, rules: list[GitIgnoreRule], *, is_dir: bool) -> bool:
    rel = path.relative_to(base).as_posix()

    # Check manual EXCLUDED_PATHS first
    for ex_path in EXCLUDED_PATHS:
        if rel == ex_path or rel.startswith(ex_path + "/"):
            return True

    ignored = False
    for rule in rules:
        if _match_rule(rel, is_dir, rule):
            ignored = not rule.negated
    return ignored


def has_negated_descendant(path: Path, base: Path, rules: list[GitIgnoreRule]) -> bool:
    rel = path.relative_to(base).as_posix().rstrip("/") + "/"
    for rule in rules:
        if not rule.negated:
            continue
        rule_path = rule.pattern.rstrip("/") + "/"
        if rule_path.startswith(rel):
            return True
    return False


def export_to_markdown(source_dir: Path, output_md: Path) -> None:
    source_dir = source_dir.resolve()
    gitignore_rules = load_gitignore(source_dir)

    with output_md.open("w", encoding="utf-8") as f:
        f.write(f"# Project Export: {source_dir.name}\n\n")

        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)

            filtered_dirs = []
            for d in dirs:
                dir_path = root_path / d

                # Check if this directory should be ignored
                ignored = is_ignored(dir_path, source_dir, gitignore_rules, is_dir=True)

                # If ignored, only keep walking if there's a !negation rule inside it
                if ignored and not has_negated_descendant(dir_path, source_dir, gitignore_rules):
                    continue

                filtered_dirs.append(d)

            # Update dirs in-place to prune the walk
            dirs[:] = filtered_dirs

            for file in files:
                file_path = root_path / file
                if is_ignored(file_path, source_dir, gitignore_rules, is_dir=False):
                    continue

                rel_path = file_path.relative_to(source_dir)

                # Skip the output file itself
                if file_path == output_md.resolve():
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")

                    f.write(f"## File: {rel_path}\n")
                    ext = file_path.suffix.lstrip(".")
                    f.write(f"```{ext}\n")
                    f.write(content)
                    f.write("\n```\n\n")
                except Exception as e:
                    f.write(f"## File: {rel_path}\n")
                    f.write(f"> Error reading file: {e}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate folder into Markdown, excluding specific paths and honoring .gitignore"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=".",
        help="Folder to process (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output markdown file name",
    )

    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    output_md = Path(args.output) if args.output else source_dir.parent / f"{source_dir.name}.md"

    try:
        export_to_markdown(source_dir, output_md)
        print(f"Created: {output_md}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
