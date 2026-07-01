import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.agent.tools import AgentTool
    from rag_engine.llm import LLM
    from rag_engine.models import SearchResult

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 4
SLEEP_BETWEEN_CALLS = 8  # seconds — avoids LLM rate-limit errors


@dataclass
class _ToolCall:
    tool: str
    query: str
    reasoning: str


@dataclass
class IterationRecord:
    step: int
    tool: str
    query: str
    reasoning: str
    found: int
    new_unique: int


@dataclass
class AgentRun:
    query: str
    iterations: list[IterationRecord] = field(default_factory=list)
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""


def _extract_json_object(text: str) -> str | None:
    """Extract the first complete {...} block, correctly handling nesting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _pick_tool(
    query: str, history: list[IterationRecord], all_results: list[SearchResult], tools: dict[str, AgentTool], llm: LLM
) -> _ToolCall:
    tool_descriptions = "\n".join(f"  - {name}: {tool.description}" for name, tool in tools.items())

    history_lines = (
        "\n".join(
            f"  Step {r.step}: {r.tool}('{r.query}') → {r.found} hits, {r.new_unique} new. Reason: {r.reasoning}"
            for r in history
        )
        if history
        else "  (none yet)"
    )

    results_preview = (
        "\n".join(f"  [{i}] {r['sec_title']}" for i, r in enumerate(all_results[:8], start=1))
        if all_results
        else "  (none yet)"
    )

    template = llm.load_prompt("agent/pick_tool")
    prompt = template.format(
        query=query,
        tool_descriptions=tool_descriptions,
        history_lines=history_lines,
        result_count=len(all_results),
        results_preview=results_preview,
    )

    try:
        raw = llm.generate(prompt, temperature=0.0)
    except ValueError as exc:
        logger.warning("LLM call failed (%s) — stopping early.", exc)
        return _ToolCall(tool="done", query="", reasoning="LLM call failed")
    logger.debug("Tool-picker raw response: %r", raw)

    raw_json = _extract_json_object(raw)
    if not raw_json:
        logger.warning("LLM returned non-JSON (%r) — stopping early.", raw[:120])
        return _ToolCall(tool="done", query="", reasoning="LLM returned unparseable response")

    try:
        raw_json = raw_json.replace("\\'", "'")
        data = json.loads(raw_json)
        return _ToolCall(
            tool=data.get("tool", "done"),
            query=data.get("args", {}).get("query", query),
            reasoning=data.get("reasoning", ""),
        )
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed (%s) — stopping early.", exc)
        return _ToolCall(tool="done", query="", reasoning="JSON parse error")


def _generate_answer(query: str, results: list[SearchResult], llm: LLM) -> str:
    if not results:
        return "I couldn't find any relevant filing excerpts for your query."

    docs = "\n\n".join(
        f"[{i}] {r['company_name']} {r['filing_type']} — {r['sec_title']}\n{r['content']}"
        for i, r in enumerate(results, start=1)
    )

    template = llm.load_prompt("agent/generate")
    prompt = template.format(query=query, docs=docs)

    try:
        return llm.generate(prompt, temperature=0.3)
    except ValueError as exc:
        logger.warning("LLM generate failed (%s).", exc)
        return "Answer could not be generated due to an LLM error."


# ------------------------------------------------------------------ #
# Public API                                                         #
# ------------------------------------------------------------------ #


async def run_agent(
    query: str, tools: dict[str, AgentTool], llm: LLM, limit: int = 5, max_iterations: int = MAX_ITERATIONS
) -> AgentRun:
    """Run agentic search loop, return populated AgentRun."""
    run = AgentRun(query=query)
    seen_ids: set[str] = set()

    for step in range(1, max_iterations + 1):
        tool_call = _pick_tool(query, run.iterations, run.results, tools, llm)

        logger.debug(
            "Step %d → tool=%r query=%r | %s",
            step,
            tool_call.tool,
            tool_call.query,
            tool_call.reasoning,
        )

        if tool_call.tool == "done" or tool_call.tool not in tools:
            logger.debug("Agent finished after %d step(s).", step - 1)
            break

        tool = tools[tool_call.tool]
        raw_results = await tool.fn(tool_call.query)

        new_results = [r for r in raw_results if r["chunk_id"] not in seen_ids]
        for r in new_results:
            seen_ids.add(r["chunk_id"])
        run.results.extend(new_results)

        run.iterations.append(
            IterationRecord(
                step=step,
                tool=tool_call.tool,
                query=tool_call.query,
                reasoning=tool_call.reasoning,
                found=len(raw_results),
                new_unique=len(new_results),
            )
        )

        if step < max_iterations:
            logger.debug("Sleeping %ds before next LLM call.", SLEEP_BETWEEN_CALLS)
            time.sleep(SLEEP_BETWEEN_CALLS)

    run.results = run.results[:limit]
    run.answer = _generate_answer(query, run.results, llm)
    return run
