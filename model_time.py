"""Format model-facing times while retaining original archive timestamps."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime

ISO_TIME = re.compile(
    r"(?<![\w/])(?P<stamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    r"(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2}))"
    r"(?:\s*[（(](?:Asia/Shanghai|UTC|GMT)(?:[+-]\d{1,2}(?::\d{2})?)?[）)])?"
)
ZONE_LABEL = re.compile(r"[；;]\s*时区[：:]\s*[\w/+:-]+")
EN_ZONE_LABEL = re.compile(r",\s*Timezone:\s*[\w/+:-]+")
LOCAL_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} 机器人\] ")
TIME_FIELDS = {"timestamp", "received", "created", "started", "ended"}


def local_time(value, zone):
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value, zone)
    elif isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is not None:
        value = value.astimezone(zone)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def model_times(value, zone, key=""):
    """Convert date values in detached model data, leaving binary URLs intact."""
    if isinstance(value, dict):
        return {k: model_times(v, zone, k) for k, v in value.items()}
    if isinstance(value, list):
        return [model_times(item, zone, key) for item in value]
    if type(value) in (int, float) and key in TIME_FIELDS and value > 0:
        return local_time(value, zone)
    if not isinstance(value, str) or key in {"url", "data"}:
        return value
    if key == "arguments":
        try:
            return json.dumps(model_times(json.loads(value), zone), ensure_ascii=False)
        except (ValueError, TypeError):
            return value

    def replace(match):
        try:
            return local_time(match.group("stamp"), zone)
        except ValueError:
            return match.group()

    return EN_ZONE_LABEL.sub("", ZONE_LABEL.sub("", ISO_TIME.sub(replace, value)))


def history_message(message, zone):
    """Keep assistant history as reply text; time remains in archive metadata."""
    message = model_times(copy.deepcopy(message), zone)
    if message.get("role") != "assistant":
        return message
    content = message.get("content")
    if isinstance(content, str):
        while match := LOCAL_STAMP.match(content):
            content = content[match.end() :]
        message["content"] = content
    elif isinstance(content, list):
        for part in content:
            if part.get("type") == "text" and part.get("text", "").strip():
                while match := LOCAL_STAMP.match(part["text"]):
                    part["text"] = part["text"][match.end() :]
                if part["text"].strip():
                    break
    return message
