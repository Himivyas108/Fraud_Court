"""
Investigative tool handlers. Each tool reveals one slice of a Case's
hidden evidence. No AI here - this is pure state mutation, same as the
calibration grader, by design (see README "where AI is NOT used").
"""
from __future__ import annotations
from server.case_generator import Case, EVIDENCE_TOOLS


class ToolError(Exception):
    pass


def call_tool(case: Case, tool_name: str, already_revealed: dict) -> dict:
    if tool_name not in EVIDENCE_TOOLS:
        raise ToolError(f"Unknown investigative tool: {tool_name}")
    if tool_name in already_revealed:
        # idempotent no-op on duplicate calls, per the edge-case spec
        return already_revealed[tool_name]
    evidence = case.hidden_evidence[tool_name]
    return evidence
