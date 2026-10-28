"""Move effective tool documentation into a stable system block."""

from __future__ import annotations

import copy
import json

from astrbot.core.agent.tool import ToolSet


def strip_descriptions(schema):
    """Remove schema annotations without touching properties or literal data."""
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    result = copy.deepcopy(schema)
    result.pop("description", None)
    for key in (
        "properties",
        "$defs",
        "definitions",
        "patternProperties",
        "dependentSchemas",
    ):
        if isinstance(result.get(key), dict):
            result[key] = {
                name: strip_descriptions(child) for name, child in result[key].items()
            }
    for key in (
        "items",
        "additionalProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedProperties",
        "unevaluatedItems",
        "anyOf",
        "oneOf",
        "allOf",
        "prefixItems",
        "additionalItems",
        "contentSchema",
    ):
        child = result.get(key)
        if isinstance(child, dict):
            result[key] = strip_descriptions(child)
        elif isinstance(child, list):
            result[key] = [strip_descriptions(item) for item in child]
    # Draft-07 dependencies may contain either a schema or property-name lists.
    if isinstance(result.get("dependencies"), dict):
        result["dependencies"] = {
            name: strip_descriptions(child)
            for name, child in result["dependencies"].items()
        }
    return result


def schemas(tools):
    if not tools:
        return []
    active = ToolSet([tool for tool in tools.tools if tool.active])
    # Use the native serializer so the tool-description editor is applied first.
    return [entry["function"] for entry in active.openai_schema()]


def render_instructions(definitions):
    parts = []
    for definition in sorted(definitions, key=lambda item: item["name"]):
        parts.append(
            "## "
            + definition["name"]
            + "\n"
            + definition.get("description", "")
            + "\n参数（JSON Schema）：\n"
            + json.dumps(
                definition.get("parameters", {}),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
    return "# 工具使用说明\n\n" + "\n\n".join(parts) if parts else ""


def tool_view(available, catalog):
    """Detach provider schemas; keep the runner's executable tools untouched."""
    definitions = schemas(catalog)
    instructions = render_instructions(definitions)
    if available is None:
        return instructions, None
    call_definitions = definitions if available is catalog else schemas(available)
    by_name = {item["name"]: item for item in call_definitions}
    tools = []
    for source in sorted(available.tools, key=lambda tool: tool.name):
        if not source.active:
            continue
        tool = copy.copy(source)
        tool.description = ""
        tool.parameters = strip_descriptions(by_name[tool.name].get("parameters", {}))
        # The editor must not reapply descriptions when providers serialize again.
        tool._lcm_schema_frozen = True
        tools.append(tool)
    return instructions, ToolSet(tools)
