from __future__ import annotations

import re
from pathlib import Path

from .schema import CorpusChunk


HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def load_markdown_corpus(corpus_dir: Path) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(path, text))
    return chunks


def chunk_markdown(path: Path, text: str) -> list[CorpusChunk]:
    title = path.stem.replace("_", " ").title()
    sections: list[tuple[str, str]] = []
    current_title = title
    buffer: list[str] = []

    for line in text.splitlines():
        if HEADING_RE.match(line):
            if buffer:
                sections.extend(_paragraph_sections(current_title, "\n".join(buffer)))
                buffer = []
            current_title = HEADING_RE.sub("", line).strip()
        else:
            buffer.append(line)
    if buffer:
        sections.extend(_paragraph_sections(current_title, "\n".join(buffer)))

    chunks = []
    for index, (section_title, body) in enumerate(sections, start=1):
        body = body.strip()
        if not body:
            continue
        chunk_id = f"{path.stem}:{index}"
        chunks.append(CorpusChunk(id=chunk_id, source=path.name, title=section_title, text=body))
    return chunks


def _paragraph_sections(title: str, body: str) -> list[tuple[str, str]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    return [(title, paragraph) for paragraph in paragraphs]
