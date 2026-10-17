from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import DEFAULT_SKILL_ROOT


@dataclass
class SkillDocument:
    name: str
    description: str
    path: Path
    body: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
        }


def _parse_frontmatter(text: str) -> tuple[Dict[str, str], str]:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    frontmatter: Dict[str, str] = {}
    end_idx = None
    for idx in range(1, len(lines)):
        line = lines[idx].strip()
        if line == "---":
            end_idx = idx
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    if end_idx is None:
        return {}, text
    body = "\n".join(lines[end_idx + 1 :]).strip()
    return frontmatter, body


def _discover_skill_files(skill_root: Path) -> List[Path]:
    if not skill_root.exists():
        return []
    return sorted(
        path
        for path in skill_root.glob("*/SKILL.md")
        if path.is_file()
    )


def load_skill_documents(
    skill_names: Optional[Iterable[str]] = None,
    *,
    skill_root: Optional[Path] = None,
) -> List[SkillDocument]:
    root = skill_root or DEFAULT_SKILL_ROOT
    requested = {name.strip() for name in skill_names or [] if name and name.strip()}
    docs: List[SkillDocument] = []
    for path in _discover_skill_files(root):
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        name = meta.get("name") or path.parent.name
        if requested and name not in requested and path.parent.name not in requested:
            continue
        docs.append(
            SkillDocument(
                name=name,
                description=meta.get("description", ""),
                path=path,
                body=body,
            )
        )
    return docs


def render_skill_context(
    skill_names: Optional[Iterable[str]] = None,
    *,
    skill_root: Optional[Path] = None,
) -> str:
    docs = load_skill_documents(skill_names, skill_root=skill_root)
    if not docs:
        return ""
    chunks: List[str] = []
    for doc in docs:
        description = f"\nDescription: {doc.description}" if doc.description else ""
        chunks.append(f"## Skill: {doc.name}{description}\n{doc.body}".strip())
    return "\n\n".join(chunks)
