"""Render human messages with one sender and one local sending time."""

from __future__ import annotations

import copy
import re

from .model_time import local_time

STAMP = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?"
BATCH_HEADER = re.compile(
    rf"^\[(?P<sender>[^\n]*\([^\n()]+\))/(?P<stamp>{STAMP})(?P<mention> @机器人)?\] ?"
)
ARCHIVE_HEADER = re.compile(
    rf"^\[(?P<stamp>{STAMP}) (?P<sender>[^\n]*\([^\n()]+\))\] ?"
)
HEADER = re.compile(
    rf"^发送人：(?P<sender>[^\n]+)\n发送时间：(?P<stamp>{STAMP}|未知)\n"
)
REMINDER = re.compile(
    r"<system_reminder>((?:(?!</?system_reminder>).)*)</system_reminder>\s*$",
    re.DOTALL,
)
IDENTITY = re.compile(r"^User ID: (.*?), Nickname: (.*)$")
CLOCK = re.compile(rf"^Current datetime: (?P<stamp>{STAMP})(?:,.*)?$")
SENDER_ID = re.compile(r"^(.*)\(([A-Za-z0-9_.:@-]+)\)$")


def sender_name(value):
    value = str(value or "")
    match = SENDER_ID.fullmatch(value)
    return ((match[1] or match[2]) if match else value).replace("\n", " ")


def user_message(message, zone, *, sender="", timestamp=None):
    """Normalize known metadata only; keep user text and all media parts."""
    result = copy.deepcopy(message)
    if result.get("role") != "user":
        return result
    content = result.get("content")
    structured = isinstance(content, list)
    if not structured and not isinstance(content, str):
        return result
    parts = content if structured else [{"type": "text", "text": content}]
    found_sender, found_time, mentioned = "", None, False
    first_text = True
    cleaned = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "text":
            cleaned.append(part)
            continue
        text = part.get("text", "")
        if first_text and text:
            first_text = False
            while match := (
                HEADER.match(text) or BATCH_HEADER.match(text) or ARCHIVE_HEADER.match(text)
            ):
                found_sender = found_sender or (
                    match["sender"] if match.re is HEADER else sender_name(match["sender"])
                )
                found_time = found_time or match["stamp"]
                mentioned = mentioned or bool(match.groupdict().get("mention"))
                text = text[match.end() :]

        def remove_reminder(match):
            nonlocal found_sender, found_time
            lines = match[1].splitlines()
            if not lines or not all(
                line.startswith(("User ID:", "Current datetime:", "Group name:"))
                for line in lines
            ):
                return match[0]
            for line in lines:
                if identity := IDENTITY.fullmatch(line):
                    found_sender = found_sender or identity[2] or identity[1]
                elif clock := CLOCK.fullmatch(line):
                    found_time = found_time or clock["stamp"]
            return ""

        without_reminder = REMINDER.sub(remove_reminder, text)
        # A native reminder may occupy its own part or follow the body on a line.
        if without_reminder != text:
            text = without_reminder.removesuffix("\n")
        if text or part.get("_no_save"):
            cleaned.append({**part, "text": text})

    name = str(sender or found_sender).replace("\n", " ")
    if not name:
        return result
    stamp = timestamp if timestamp is not None else found_time
    try:
        date = local_time(stamp, zone) if stamp and stamp != "未知" else "未知"
    except (ValueError, TypeError, AttributeError, OverflowError, OSError):
        date = "未知"
    header = f"发送人：{name}\n发送时间：{date}\n"
    if mentioned and not any(
        isinstance(part, dict) and "@机器人" in part.get("text", "")
        for part in cleaned
    ):
        header += "@机器人\n"
    if (
        cleaned
        and isinstance(cleaned[0], dict)
        and cleaned[0].get("type") == "text"
        and not cleaned[0].get("_no_save")
    ):
        cleaned[0] = {**cleaned[0], "text": header + cleaned[0]["text"]}
    else:
        cleaned.insert(0, {"type": "text", "text": header})
    result["content"] = cleaned if structured else cleaned[0]["text"]
    return result


def event_message(message, event, zone):
    name = getattr(event, "get_sender_name", lambda: "")() or getattr(
        event, "get_sender_id", lambda: ""
    )()
    stamp = getattr(getattr(event, "message_obj", None), "timestamp", None)
    return user_message(message, zone, sender=name, timestamp=stamp or None)


def node_message(message, node, zone):
    """Use actual archived event time when available, never import time."""
    sender = node.get("sender", "")
    extra = node.get("payload", {}).get("extra", {})
    human = sender and sender != "机器人" and not node.get("level")
    return user_message(
        message,
        zone,
        sender=sender_name(sender) if human else "",
        timestamp=(
            node.get("timestamp") if human and not extra.get("time_unknown") else None
        ),
    )
