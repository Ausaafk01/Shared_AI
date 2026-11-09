"""
Formatting and rendering for Context Builder.
"""

from typing import Any, Sequence

from ai_brain.context.models import ContextItem, ContextSection


class ContextFormatter:
    """
    Renders selected context items into structured, agent-ready markdown and plain text.
    
    Guarantees:
    - Clear distinction between established facts, decisions, tasks, and disputed claims.
    - Prominent warning banners for disputed knowledge to prevent hallucination.
    - Clean grouping into logical sections.
    - Transparent provenance attribution.
    """

    def __init__(self, *, include_provenance_section: bool = True) -> None:
        self.include_provenance_section = include_provenance_section

    def format_item_line(self, item: ContextItem) -> str:
        """Format a single item as a bullet point."""
        agents = ", ".join(item.source_agent_ids) if item.source_agent_ids else "system"
        if item.is_disputed or item.status == "DISPUTED":
            return f"- ⚠️ [DISPUTED] {item.content} (ID: `{item.memory_id}`, Source Agents: {agents})"
        elif item.memory_type == "DECISION":
            return f"- [DECISION] {item.content} (ID: `{item.memory_id}`)"
        elif item.memory_type == "TASK":
            return f"- [ ] [TASK] {item.content} (ID: `{item.memory_id}`)"
        elif item.memory_type in ("FACT", "CODE_KNOWLEDGE"):
            return f"- [FACT] {item.content} (ID: `{item.memory_id}`)"
        else:
            return f"- [{item.memory_type}] {item.content} (ID: `{item.memory_id}`)"

    def build_sections(
        self,
        items: Sequence[ContextItem],
    ) -> list[ContextSection]:
        """
        Partition context items into logical domain sections.
        """
        decisions: list[ContextItem] = []
        tasks: list[ContextItem] = []
        facts: list[ContextItem] = []
        disputed: list[ContextItem] = []
        relationships: list[ContextItem] = []

        for item in items:
            if item.is_disputed or item.status == "DISPUTED":
                disputed.append(item)
            elif item.memory_type == "DECISION":
                decisions.append(item)
            elif item.memory_type == "TASK":
                tasks.append(item)
            else:
                facts.append(item)

            if getattr(item, "graph_relationships", None):
                relationships.append(item)

        sections: list[ContextSection] = []

        if decisions:
            sections.append(ContextSection(name="Active Decisions & Constraints", items=tuple(decisions)))
        if tasks:
            sections.append(ContextSection(name="Active Tasks", items=tuple(tasks)))
        if facts:
            sections.append(ContextSection(name="Established Facts & Architecture", items=tuple(facts)))
        if disputed:
            sections.append(ContextSection(name="Disputed Knowledge", items=tuple(disputed)))
        if relationships:
            sections.append(ContextSection(name="Connected Knowledge Graph", items=tuple(relationships)))

        return sections

    def render_markdown(
        self,
        *,
        project_id: str,
        query: str,
        sections: Sequence[ContextSection],
        items: Sequence[ContextItem],
        excluded_items: Sequence[Any] = (),
    ) -> str:
        """
        Render a full agent-ready Markdown document.
        """
        lines: list[str] = []
        lines.append(f"# Project Knowledge Context: `{project_id}`")
        if query:
            lines.append(f"**Query Focus:** *{query}*\n")
        else:
            lines.append("")

        # Disputed knowledge omission notice
        omitted_disputes = [
            ex for ex in excluded_items
            if getattr(ex, "is_disputed", False) or getattr(ex, "status", "") == "DISPUTED"
        ]
        if omitted_disputes:
            count = len(omitted_disputes)
            disp_ids = ", ".join(f"`{ex.memory_id}`" for ex in omitted_disputes)
            lines.append(
                f"> ⚠️ **DISPUTED KNOWLEDGE OMITTED:** {count} potentially relevant disputed claim(s) "
                f"({disp_ids}) were omitted due to token budget constraints. "
                f"Unresolved contradictions exist regarding this topic; do not assume completeness.\n"
            )

        if not items:
            lines.append("*No relevant memories found for this project and query.*\n")
            return "\n".join(lines).strip()

        for sec in sections:
            lines.append(f"## {sec.name}")
            if sec.name == "Disputed Knowledge":
                lines.append(
                    "> ⚠️ **ATTENTION AGENT:** The following items represent conflicting or disputed "
                    "information. Do NOT treat them as established ground truth. Evaluate alternative "
                    "perspectives or request clarification if necessary."
                )
            for item in sec.items:
                lines.append(self.format_item_line(item))
            lines.append("")

        if self.include_provenance_section and items:
            lines.append("## Provenance & Attribution Summary")
            lines.append("| Memory ID | Type | Status | Source Agents | Source Events |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for item in items:
                ag_str = ", ".join(item.source_agent_ids) if item.source_agent_ids else "N/A"
                ev_str = ", ".join(item.source_event_ids) if item.source_event_ids else "N/A"
                lines.append(f"| `{item.memory_id}` | {item.memory_type} | {item.status} | {ag_str} | {ev_str} |")
            lines.append("")

        return "\n".join(lines).strip()

    def render_text(
        self,
        *,
        project_id: str,
        query: str,
        sections: Sequence[ContextSection],
        items: Sequence[ContextItem],
        excluded_items: Sequence[Any] = (),
    ) -> str:
        """
        Render a plain text representation.
        """
        lines: list[str] = []
        lines.append(f"PROJECT CONTEXT: {project_id}")
        if query:
            lines.append(f"QUERY: {query}")
        lines.append("=" * 40)

        omitted_disputes = [
            ex for ex in excluded_items
            if getattr(ex, "is_disputed", False) or getattr(ex, "status", "") == "DISPUTED"
        ]
        if omitted_disputes:
            disp_ids = ", ".join(ex.memory_id for ex in omitted_disputes)
            lines.append(f"[NOTICE: {len(omitted_disputes)} disputed claim(s) ({disp_ids}) omitted due to token budget constraints.]\n")

        if not items:
            lines.append("No relevant memories found.")
            return "\n".join(lines).strip()

        for sec in sections:
            lines.append(f"\n[{sec.name.upper()}]")
            for item in sec.items:
                disp = " [DISPUTED]" if item.is_disputed or item.status == "DISPUTED" else ""
                lines.append(f" - ({item.memory_type}{disp}) {item.content}")

        return "\n".join(lines).strip()

