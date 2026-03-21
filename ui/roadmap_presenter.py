"""
Roadmap presenter: format Roadmap or RoadmapSnapshot to Markdown

Key features:
- format_roadmap_markdown(): Full narrative text for chat bubble
"""
from __future__ import annotations

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from domain import Roadmap
    from domain.snapshots import RoadmapSnapshot

def format_roadmap_markdown(roadmap: Union[Roadmap, RoadmapSnapshot]) -> str:
    """
    Format Roadmap as full narrative Markdown text

    Output format:
    - Topic and duration header
    - Optional description
    - For each milestone:
        Week number and topic
        Description
        Learning objectives
        Resources with URLs

    Args:
        roadmap: Roadmap or RoadmapSnapshot domain object
        
    Returns:
        Formatted Markdown string for display in chat bubble
    """
    lines = [
        f"**{roadmap.topic}** ({roadmap.duration_week} tuần)",
        "",
    ]
    if roadmap.description:
        lines.append(roadmap.description.strip())
        lines.append("")

    for m in roadmap.milestones:
        lines.append(f"**Tuần {m.week}: {m.topic}**")
        lines.append("")
        lines.append(m.description)
        if m.learning_objectives:
            lines.append("")
            lines.append("*Mục tiêu:*")
            for obj in m.learning_objectives:
                lines.append(f"- {obj}")
        lines.append("")
        lines.append("*Tài liệu:*")
        for res in m.resources:
            lines.append(f"- [{res.title}]({res.url}) ({res.type})")
        lines.append("")

    return "\n".join(lines).strip()