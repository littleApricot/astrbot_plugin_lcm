"""Keep media bytes separate from immutable message payloads."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import shutil
from pathlib import Path
from uuid import uuid4


class MediaArchive:
    def __init__(self, store):
        self.store = store
        self.root = store.path.parent / "media"
        self.root.mkdir(exist_ok=True)

    async def capture(self, scope, node, component):
        ident = "a_" + uuid4().hex
        original = str(
            getattr(component, "url", "")
            or getattr(component, "file", "")
            or getattr(component, "path", "")
        )
        with self.store.connection(True) as db:
            existing = db.execute(
                "SELECT id FROM media WHERE scope=? AND node=? AND original=?",
                (scope, node, original),
            ).fetchone()
            if existing:
                return existing[0]
            db.execute(
                "INSERT INTO media(id,scope,node,original,status) VALUES(?,?,?,?,'pending')",
                (ident, scope, node, original),
            )
        try:
            resolver = getattr(component, "convert_to_file_path", None) or getattr(
                component, "get_file", None
            )
            if not resolver:
                raise ValueError("media_resolver_unavailable")
            source = Path(await resolver())
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            suffix = source.suffix if len(source.suffix) <= 12 else ""
            dest = self.root / (digest + suffix)
            if not dest.exists():
                shutil.copyfile(source, dest)
            mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            with self.store.connection(True) as db:
                db.execute(
                    "UPDATE media SET path=?,mime=?,sha256=?,status='ready' WHERE id=?",
                    (str(dest), mime, digest, ident),
                )
        except Exception as exc:
            with self.store.connection(True) as db:
                db.execute(
                    "UPDATE media SET status='missing',error=? WHERE id=?",
                    (type(exc).__name__, ident),
                )
        return ident

    def records(self, scope, node):
        with self.store.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM media WHERE scope=? AND node=?", (scope, node)
                )
            ]

    def enrich(self, scope, result):
        """Attach discoverable archive assets to tool and viewer results."""
        nodes = list(result.get("records", []))
        if isinstance(result.get("node"), dict):
            nodes.append(result["node"])
        for unit in result.get("units", []):
            nodes.extend(unit["records"])
        for node in nodes:
            node["media"] = self.records(scope, node["id"])

    def data_url(self, scope, ident):
        result = self.store.expand(scope, ident).get("media")
        if not result or result["status"] != "ready":
            raise ValueError("media_not_available")
        path = Path(result["path"])
        if self.root.resolve() not in path.resolve().parents:
            raise ValueError("invalid_media_path")
        return {
            **result,
            "data_url": f"data:{result['mime']};base64,"
            + base64.b64encode(path.read_bytes()).decode(),
        }
