"""Build bounded visual inputs without changing archived or native history."""

from __future__ import annotations

import copy
import re

from .image_tokens import ImageTokenEstimator

DATA_IMAGE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+")


class ImageContext:
    """Resolve existing gallery metadata within one detached projection."""

    def __init__(self, owner):
        lookup = getattr(getattr(owner, "context", None), "get_registered_star", None)
        star = lookup("astrbot_plugin_meme_library") if callable(lookup) else None
        plugin = getattr(star, "star_cls", None)
        self.resolver = getattr(plugin, "decision_image_description", None)
        self.descriptions = {}
        self.owner = owner

    async def description(self, reference):
        if isinstance(reference, dict):
            reference = reference.get("url", "")
        if not isinstance(reference, str):
            return "图片"
        if reference not in self.descriptions:
            value = ""
            if callable(self.resolver):
                try:
                    value = await self.resolver(reference)
                except Exception:
                    # Missing or unavailable metadata must not block a reply.
                    pass
            self.descriptions[reference] = value or "图片"
        return self.descriptions[reference]

    async def text_only(self, value):
        """Replace image blocks and embedded image bytes in a detached value."""
        if isinstance(value, dict):
            if value.get("type") in ("image_url", "input_image"):
                return {
                    "type": "text",
                    "text": await self.description(value.get("image_url", "")),
                }
            return {key: await self.text_only(item) for key, item in value.items()}
        if isinstance(value, list):
            return [await self.text_only(item) for item in value]
        if isinstance(value, str) and "data:image/" in value:
            parts, offset = [], 0
            for match in DATA_IMAGE.finditer(value):
                parts.extend(
                    (value[offset : match.start()], await self.description(match[0]))
                )
                offset = match.end()
            return "".join(parts) + value[offset:]
        return value

    async def history(self, messages, keep, nodes=(), *, token_budget=None, model=""):
        """Limit only the supplied history; callers keep current input separate."""
        ordered, runs = [], set()
        for node in nodes:
            run = node["round_id"]
            if run:
                if run in runs:
                    continue
                runs.add(run)
                ordered.extend(item for item in nodes if item["round_id"] == run)
            else:
                ordered.append(node)
        values = [
            copy.deepcopy(message)
            if isinstance(message, dict)
            else message.model_dump(mode="json")
            for message in messages
        ]
        images = [
            part
            for message in values
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        if token_budget is None:
            retained = min(len(images), max(0, keep))
            self.stats = {"mode": "count", "limit": max(0, keep)}
        else:
            token_budget = max(0, token_budget)
            estimator = ImageTokenEstimator(model)
            retained, used = 0, 0
            for part in reversed(images):
                cost = estimator.count(part)
                if used + cost > token_budget:
                    break
                retained += 1
                used += cost
            self.stats = {
                "mode": "tokens",
                "budget": token_budget,
                "estimated_tokens": used,
                "estimator": estimator.method,
            }
        self.stats.update(total_images=len(images), retained_images=retained)
        replace = len(images) - retained
        seen, node_index = 0, 0
        for message in values:
            node = None
            if message.get("role") != "system":
                if node_index < len(ordered):
                    node = ordered[node_index]
                node_index += 1
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = await self.text_only(content)
                continue
            if not isinstance(content, list):
                continue
            has_image = False
            for index, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                has_image = True
                seen += 1
                if seen <= replace:
                    content[index] = {
                        "type": "text",
                        "text": await self.description(part.get("image_url", "")),
                    }
            if has_image and node:
                media = self.owner.media.records(node["scope"], node["id"])
                ids = [item["id"] for item in media]
                content.append(
                    {
                        "type": "text",
                        "text": "[历史消息 id="
                        + node["id"]
                        + ("；媒体 id=" + ",".join(ids) if ids else "")
                        + "]",
                    }
                )
        # Summaries also sanitize serialized tool results and interrupted records.
        if retained == 0:
            values = await self.text_only(values)
        return [
            value
            if isinstance(original, dict)
            else type(original).model_validate(value)
            for original, value in zip(messages, values)
        ]
