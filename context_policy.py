"""Compose declared model views without rewriting archived messages."""

from __future__ import annotations

import copy
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from astrbot.core.astr_main_agent_resources import (
    TOOL_CALL_PROMPT,
    TOOL_CALL_PROMPT_SKILLS_LIKE_MODE,
)

from .runtime_clock import runtime_time
from .user_message import event_message

BLOCKED_TOOLS = {"oauth-status", "oauth-logout", "protect-page"}
SYSTEM_START = "[系统提示｜来源：{owner}]"
TEMP_START = "[本轮临时提示｜来源：{owner}]"
DISABLED_SKILLS = re.compile(
    r"## Skills\b.*?User has not enabled the Computer Use feature\."
    r".*?AstrBot WebUI -> Config\.",
    re.DOTALL,
)


def wrap(owner, text, temporary=False):
    start = TEMP_START if temporary else SYSTEM_START
    end = "[/本轮临时提示]" if temporary else "[/系统提示]"
    return start.format(owner=owner) + "\n" + text.strip() + "\n" + end


def native(message):
    return (
        copy.deepcopy(message)
        if isinstance(message, dict)
        else message.model_dump(mode="json")
    )


class ContextRules:
    """Discover plugin declarations and apply the same history view everywhere."""

    def __init__(self, owner):
        self.owner = owner

    def declarations(self):
        context = self.owner.context
        get_all = getattr(context, "get_all_stars", None)
        stars = get_all() if callable(get_all) else []
        if not stars:
            lookup = getattr(context, "get_registered_star", None)
            if callable(lookup):
                stars = [
                    lookup(name)
                    for name in (
                        "astrbot_plugin_self_prompt",
                        "astrbot_plugin_meme_library",
                        "astrbot_plugin_silence",
                    )
                ]
        declarations, seen = [], set()
        for star in stars:
            plugin = getattr(star, "star_cls", None)
            if not plugin or not getattr(star, "activated", True):
                continue
            if not getattr(plugin, "config", {}).get("enabled", True):
                continue
            declare = getattr(plugin, "lcm_context_rules", None)
            if not callable(declare):
                continue
            value = declare()
            ident = value.get("owner")
            if not isinstance(ident, str) or not ident or ident in seen:
                continue
            policies = value.get("history_tools", {})
            if not isinstance(policies, dict) or any(
                policy not in {"keep", "turn", "attachment"}
                for policy in policies.values()
            ):
                raise ValueError("插件历史声明包含无效策略")
            if any(policy != "keep" for policy in policies.values()) and not callable(
                value.get("history_projector")
            ):
                raise ValueError("临时或转换历史策略必须声明历史投影函数")
            seen.add(ident)
            declarations.append(value)
        return declarations

    def history_entries(self, messages):
        entries = [(index, native(message)) for index, message in enumerate(messages)]
        for declaration in self.declarations():
            project = declaration.get("history_projector")
            if not callable(project):
                continue
            projected = project([message for _, message in entries])
            last = -1
            updated = []
            for index, message in projected:
                if type(index) is not int or not last < index < len(entries):
                    raise ValueError("插件历史投影必须保留原顺序及来源索引")
                last = index
                updated.append((entries[index][0], native(message)))
            entries = updated
        return entries

    def project_history(self, messages):
        return [message for _, message in self.history_entries(messages)]

    def configure_request(self, event, req):
        # Put sender/event time on the message; current time stays in runtime hints.
        kept = []
        for part in req.extra_user_content_parts or []:
            value = part if isinstance(part, dict) else part.model_dump()
            text = value.get("text", "")
            if (
                value.get("type") == "text"
                and text.startswith("<system_reminder>")
                and text.endswith("</system_reminder>")
            ):
                body = text[len("<system_reminder>") : -len("</system_reminder>")]
                lines = body.splitlines()
                if lines and all(
                    line.startswith(("User ID:", "Group name:", "Current datetime:"))
                    for line in lines
                ):
                    continue
            kept.append(part)
        req.extra_user_content_parts = kept
        config = getattr(self.owner.context, "get_config", lambda **kwargs: {})(
            umo=event.unified_msg_origin
        )
        zone = ZoneInfo(config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
        req.prompt = event_message(
            {"role": "user", "content": req.prompt or ""}, event, zone
        )["content"]
        req.system_prompt = self.system(event, req.system_prompt or "")
        event.set_extra("lcm_request_system", req.system_prompt)
        event.set_extra(
            "lcm_request_system_base", event.get_extra("lcm_system_base", "")
        )
        if req.func_tool is not None:
            req.func_tool = copy.copy(req.func_tool)
            req.func_tool.tools = [
                tool for tool in req.func_tool.tools if tool.name not in BLOCKED_TOOLS
            ]

    def system(self, event, raw, tool_instructions=""):
        previous = event.get_extra("lcm_rendered_system", "")
        base = event.get_extra("lcm_system_base", "")
        if previous and previous in raw:
            raw = base + raw.replace(previous, "", 1)
        else:
            # The runner retains the initial system message. Later provider calls
            # must refresh that rendering, even after tools changed the catalog.
            initial = event.get_extra("lcm_request_system", "")
            if initial and initial in raw:
                raw = event.get_extra("lcm_request_system_base", "") + raw.replace(
                    initial, "", 1
                )
        raw = DISABLED_SKILLS.sub("", raw)
        for prompt in (TOOL_CALL_PROMPT, TOOL_CALL_PROMPT_SKILLS_LIKE_MODE):
            raw = raw.replace(prompt, "")
        raw = raw.replace("# Persona Instructions", "").strip()
        event.set_extra("lcm_system_base", raw)
        records = []
        if raw:
            records.append(("astrbot.core/未声明来源", raw))
        if tool_instructions:
            records.append(("astrbot.tools", tool_instructions))
        for declaration in self.declarations():
            value = declaration.get("system", "")
            value = value(event) if callable(value) else value
            if value and value.strip():
                records.append((declaration["owner"], value))
        rendered = "\n\n".join(wrap(owner, text) for owner, text in records)
        event.set_extra("lcm_rendered_system", rendered)
        event.set_extra(
            "lcm_system_sources",
            [{"owner": owner, "characters": len(text)} for owner, text in records],
        )
        return rendered

    def temporary(self, event):
        config = getattr(self.owner.context, "get_config", lambda **kwargs: {})(
            umo=event.unified_msg_origin
        )
        zone = ZoneInfo(config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
        current = datetime.now(zone)
        runtime = runtime_time(current)
        records = [("astrbot.runtime", runtime)]
        for declaration in self.declarations():
            value = declaration.get("temporary", "")
            value = value(event) if callable(value) else value
            if value and value.strip():
                records.append((declaration["owner"], value))
        event.set_extra(
            "lcm_temporary_sources",
            [{"owner": owner, "characters": len(text)} for owner, text in records],
        )
        return "\n\n".join(wrap(owner, text, True) for owner, text in records)

    def manifest(self, event):
        return {
            "system": event.get_extra("lcm_system_sources", []),
            "temporary": event.get_extra("lcm_temporary_sources", []),
            "history": [
                {"owner": rule["owner"], "tools": rule.get("history_tools", {})}
                for rule in self.declarations()
            ],
        }
