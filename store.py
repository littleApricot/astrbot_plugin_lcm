"""Durable originals, summary provenance, and atomic working projections."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def estimate(value):
    """Estimate text/structure tokens while counting media independently of bytes."""
    media = 0

    def compact(item):
        nonlocal media
        if isinstance(item, list):
            return [compact(x) for x in item]
        if isinstance(item, dict):
            result = {}
            for key, val in item.items():
                if key in ("image_url", "audio_url"):
                    media += 765 if key == "image_url" else 500
                    result[key] = "[media]"
                else:
                    result[key] = compact(val)
            return result
        return item

    text = encode(compact(value))
    cjk = sum("\u4e00" <= c <= "\u9fff" for c in text)
    return max(1, int(cjk * 0.7 + (len(text) - cjk) / 3) + media)


class Store:
    """Use short WAL transactions; never hold one across an awaited model call."""

    def __init__(self, path: Path, count=estimate):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = count
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS scopes (
                    id TEXT PRIMARY KEY, origin TEXT NOT NULL, cid TEXT NOT NULL,
                    title TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 0,
                    imported INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    scope TEXT NOT NULL, level INTEGER NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, children TEXT NOT NULL DEFAULT '[]',
                    tokens INTEGER NOT NULL, lo INTEGER, hi INTEGER,
                    active INTEGER NOT NULL DEFAULT 1, round_id TEXT,
                    reservation TEXT, source_key TEXT, sender TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL, received REAL NOT NULL,
                    UNIQUE(scope, source_key)
                );
                CREATE INDEX IF NOT EXISTS nodes_view ON nodes(scope,active,lo);
                CREATE INDEX IF NOT EXISTS nodes_round ON nodes(scope,round_id,seq);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, version INTEGER NOT NULL,
                    base TEXT NOT NULL, status TEXT NOT NULL, started REAL NOT NULL,
                    ended REAL, error TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, run TEXT NOT NULL,
                    step INTEGER NOT NULL, payload TEXT NOT NULL,
                    tokens INTEGER NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS requests_run ON requests(run,step);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, level INTEGER NOT NULL,
                    sources TEXT NOT NULL, status TEXT NOT NULL, result TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0, retry REAL NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '', created REAL NOT NULL, ended REAL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_job ON jobs(scope)
                    WHERE status IN ('queued','running','ready','retry');
                CREATE TABLE IF NOT EXISTS media (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, node TEXT NOT NULL,
                    original TEXT NOT NULL, path TEXT, mime TEXT, status TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '', sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS deliveries (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, run TEXT, payload TEXT NOT NULL,
                    created REAL NOT NULL, status TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def scope(self, origin, cid, title=""):
        key = hashlib.sha256(encode([origin, cid]).encode()).hexdigest()[:24]
        with self.connection(True) as db:
            db.execute(
                "INSERT OR IGNORE INTO scopes(id,origin,cid,title,created) VALUES(?,?,?,?,?)",
                (key, origin, cid, title or origin, time.time()),
            )
        return key

    def scopes(self):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute("""SELECT s.*,
                (SELECT COUNT(*) FROM nodes n WHERE n.scope=s.id AND level=0) records,
                (SELECT COUNT(*) FROM runs r WHERE r.scope=s.id AND status='running') running
                FROM scopes s ORDER BY created DESC""")
            ]

    def node(self, scope, ident, db=None):
        if db is None:
            with self.connection() as conn:
                return self.node(scope, ident, conn)
        row = db.execute(
            "SELECT * FROM nodes WHERE scope=? AND id=?", (scope, ident)
        ).fetchone()
        if row is None:
            raise ValueError("record_not_found_in_session")
        out = dict(row)
        out["payload"] = json.loads(out["payload"])
        out["children"] = json.loads(out["children"])
        return out

    def append(
        self,
        scope,
        message,
        *,
        source=None,
        sender="",
        timestamp=None,
        round_id=None,
        extra=None,
        tokens=None,
    ):
        payload = {"message": message, "extra": extra or {}}
        with self.connection(True) as db:
            if source:
                row = db.execute(
                    "SELECT id FROM nodes WHERE scope=? AND source_key=?",
                    (scope, source),
                ).fetchone()
                if row:
                    return row[0]
            ident = "m_" + uuid4().hex
            cur = db.execute(
                """INSERT INTO nodes(id,scope,level,kind,payload,tokens,round_id,
                source_key,sender,timestamp,received) VALUES(?,?,0,'raw',?,?,?,?,?,?,?)""",
                (
                    ident,
                    scope,
                    encode(payload),
                    tokens if tokens is not None else self.count(message),
                    round_id,
                    source,
                    sender,
                    timestamp or time.time(),
                    time.time(),
                ),
            )
            db.execute("UPDATE nodes SET lo=seq,hi=seq WHERE seq=?", (cur.lastrowid,))
            db.execute("UPDATE scopes SET version=version+1 WHERE id=?", (scope,))
            if (
                round_id
                and db.execute(
                    "SELECT 1 FROM runs WHERE id=? AND status='running'", (round_id,)
                ).fetchone()
            ):
                db.execute("UPDATE nodes SET active=0 WHERE id=?", (ident,))
            return ident

    def import_history(self, scope, messages):
        """Import available legacy provider history once, before live originals."""
        with self.connection(True) as db:
            if db.execute(
                "SELECT imported FROM scopes WHERE id=?", (scope,)
            ).fetchone()[0]:
                return
            minimum = db.execute(
                "SELECT COALESCE(MIN(lo),0) FROM nodes WHERE scope=?", (scope,)
            ).fetchone()[0]
            group = "legacy_" + uuid4().hex
            for index, msg in enumerate(messages):
                if msg.get("role") not in ("user", "assistant", "tool", "system"):
                    continue
                # Legacy boundaries are inferred, never advertised as exact snapshots.
                if msg.get("role") == "user":
                    group = "legacy_" + uuid4().hex
                order = minimum - len(messages) + index
                db.execute(
                    """INSERT INTO nodes(id,scope,level,kind,payload,tokens,lo,hi,round_id,
                    timestamp,received) VALUES(?,?,0,'raw',?,?,?,?,?,?,?)""",
                    (
                        "m_" + uuid4().hex,
                        scope,
                        encode(
                            {
                                "message": msg,
                                "extra": {"legacy": True, "time_unknown": True},
                            }
                        ),
                        self.count(msg),
                        order,
                        order,
                        group,
                        time.time(),
                        time.time(),
                    ),
                )
            db.execute(
                "UPDATE scopes SET imported=1,version=version+1 WHERE id=?", (scope,)
            )

    def view(self, scope, exclude=()):
        with self.connection() as db:
            state = db.execute("SELECT * FROM scopes WHERE id=?", (scope,)).fetchone()
            if state is None:
                raise ValueError("session_not_found")
            nodes = [
                self.node(scope, r[0], db)
                for r in db.execute(
                    "SELECT id FROM nodes WHERE scope=? AND active=1 ORDER BY lo,seq",
                    (scope,),
                )
                if r[0] not in exclude
            ]
            totals = {}
            for node in nodes:
                key = str(node["level"])
                totals[key] = totals.get(key, 0) + node["tokens"]
            jobs = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM jobs WHERE scope=? ORDER BY created DESC LIMIT 8",
                    (scope,),
                )
            ]
            return {
                "scope": dict(state),
                "version": state["version"],
                "nodes": nodes,
                "tokens": totals,
                "token_method": "estimate",
                "jobs": jobs,
            }

    def reserve(self, scope, ids, owner):
        with self.connection(True) as db:
            for ident in ids:
                row = db.execute(
                    "SELECT active,reservation FROM nodes WHERE scope=? AND id=?",
                    (scope, ident),
                ).fetchone()
                if row is None or not row[0] or row[1] not in (None, owner):
                    raise ValueError("input_already_compacted_or_reserved")
                db.execute(
                    "UPDATE nodes SET reservation=? WHERE scope=? AND id=?",
                    (owner, scope, ident),
                )

    def release(self, owner):
        with self.connection(True) as db:
            db.execute(
                "UPDATE nodes SET reservation=NULL WHERE reservation=?", (owner,)
            )

    def begin(self, scope, owner, base, inputs):
        ident = "r_" + uuid4().hex
        with self.connection(True) as db:
            for mid in inputs:
                row = db.execute(
                    "SELECT active,reservation FROM nodes WHERE scope=? AND id=?",
                    (scope, mid),
                ).fetchone()
                if row is None or not row[0] or row[1] != owner:
                    raise ValueError("round_input_not_reserved")
            db.execute(
                "INSERT INTO runs(id,scope,version,base,status,started) VALUES(?,?,?,?,'running',?)",
                (
                    ident,
                    scope,
                    base["version"],
                    encode([n["id"] for n in base["nodes"]]),
                    time.time(),
                ),
            )
            for mid in inputs:
                db.execute(
                    "UPDATE nodes SET round_id=?,reservation=NULL WHERE scope=? AND id=?",
                    (ident, scope, mid),
                )
        return ident

    def finish(self, run, status="complete", error=""):
        with self.connection(True) as db:
            scope = db.execute(
                "SELECT scope FROM runs WHERE id=? AND status='running'", (run,)
            ).fetchone()
            if scope:
                db.execute(
                    "UPDATE runs SET status=?,ended=?,error=? WHERE id=?",
                    (status, time.time(), error, run),
                )
                db.execute("UPDATE nodes SET active=1 WHERE round_id=?", (run,))
                db.execute(
                    "UPDATE scopes SET version=version+1 WHERE id=?", (scope[0],)
                )

    def snapshot(self, scope, run, payload):
        with self.connection(True) as db:
            step = db.execute(
                "SELECT COALESCE(MAX(step),0)+1 FROM requests WHERE run=?", (run,)
            ).fetchone()[0]
            db.execute(
                "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
                (
                    "q_" + uuid4().hex,
                    scope,
                    run,
                    step,
                    encode(payload),
                    self.count(payload),
                    time.time(),
                ),
            )
            return step

    def runs(self, scope, cursor=0, limit=40):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM runs WHERE scope=? ORDER BY started DESC LIMIT ? OFFSET ?",
                    (scope, limit, int(cursor)),
                )
            ]

    def requests(self, scope, run):
        with self.connection() as db:
            rows = db.execute(
                "SELECT id,step,tokens,created FROM requests WHERE scope=? AND run=? ORDER BY step",
                (scope, run),
            )
            return [dict(r) for r in rows]

    def request(self, scope, ident):
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM requests WHERE scope=? AND id=?", (scope, ident)
            ).fetchone()
            if row is None:
                raise ValueError("request_not_found")
            result = dict(row)
            result["payload"] = json.loads(result["payload"])
            run = db.execute(
                "SELECT * FROM runs WHERE scope=? AND id=?", (scope, result["run"])
            ).fetchone()
            result["run_info"] = dict(run)
            return result

    def history(self, scope, cursor=0, limit=50):
        with self.connection() as db:
            rows = db.execute(
                "SELECT id FROM nodes WHERE scope=? AND level=0 ORDER BY lo DESC,seq DESC LIMIT ? OFFSET ?",
                (scope, limit + 1, int(cursor)),
            ).fetchall()
            return {
                "records": [self.node(scope, r[0], db) for r in rows[:limit]],
                "next_cursor": str(int(cursor) + limit) if len(rows) > limit else None,
            }

    def search(
        self,
        scope,
        query,
        target="both",
        sender="",
        start=0,
        end=float("inf"),
        cursor="",
        limit=20,
        ids=None,
        level=None,
    ):
        query = query.strip()
        if (
            not query
            or target not in ("raw", "summaries", "both")
            or not 1 <= limit <= 50
        ):
            raise ValueError("invalid_search")
        fingerprint = hashlib.sha256(
            encode([scope, query, target, sender, start, end, ids, level]).encode()
        ).hexdigest()[:16]
        offset = 0
        if cursor:
            fingerprint_in, offset_in = cursor.split(":")
            if fingerprint_in != fingerprint:
                raise ValueError("cursor_does_not_match_query")
            offset = max(0, int(offset_in))
        # SQLite LIKE does not case-fold Chinese/Unicode uniformly; Python casefold is literal.
        with self.connection() as db:
            rows = db.execute(
                "SELECT id FROM nodes WHERE scope=? AND timestamp>=? AND timestamp<=? ORDER BY lo DESC,seq DESC",
                (scope, start, end),
            )
            hits = []
            for row in rows:
                n = self.node(scope, row[0], db)
                if ids is not None and n["id"] not in ids:
                    continue
                if level is not None and n["level"] != level:
                    continue
                if (
                    target == "raw"
                    and n["level"] != 0
                    or target == "summaries"
                    and n["level"] == 0
                ):
                    continue
                if sender and sender.casefold() not in n["sender"].casefold():
                    continue
                text = encode(n["payload"])
                if query.casefold() in text.casefold():
                    hits.append(n)
                    if len(hits) >= offset + limit + 1:
                        break
            selected = hits[offset : offset + limit]
            return {
                "records": selected,
                "next_cursor": f"{fingerprint}:{offset + limit}"
                if len(hits) > offset + limit
                else None,
            }

    def describe(self, scope, ident="", cursor=0, limit=30):
        if not ident:
            view = self.view(scope)
            nodes = view.pop("nodes")
            return {
                **view,
                "records": nodes[int(cursor) : int(cursor) + limit],
                "next_cursor": str(int(cursor) + limit)
                if len(nodes) > int(cursor) + limit
                else None,
            }
        node = self.node(scope, ident)
        children = node["children"]
        return {
            "node": node,
            "records": [
                self.node(scope, c) for c in children[int(cursor) : int(cursor) + limit]
            ],
            "next_cursor": str(int(cursor) + limit)
            if len(children) > int(cursor) + limit
            else None,
        }

    def expand(self, scope, ident, target="original", cursor=0, limit=1):
        if target not in ("original", "children") or not 1 <= limit <= 10:
            raise ValueError("invalid_expand")
        with self.connection() as db:
            media = db.execute(
                "SELECT * FROM media WHERE id=? AND scope=?", (ident, scope)
            ).fetchone()
            if media:
                return {"media": dict(media), "next_cursor": None}
            run = db.execute(
                "SELECT * FROM runs WHERE id=? AND scope=?", (ident, scope)
            ).fetchone()
            root = None if run else self.node(scope, ident, db)
            if target == "children" and root and root["level"] > 0:
                children = root["children"]
                return {
                    "records": [
                        self.node(scope, c, db)
                        for c in children[int(cursor) : int(cursor) + limit]
                    ],
                    "next_cursor": str(int(cursor) + limit)
                    if len(children) > int(cursor) + limit
                    else None,
                }
            leaves = []
            pending = [root] if root else []
            while pending:
                n = pending.pop()
                if n["level"]:
                    pending.extend(
                        self.node(scope, c, db) for c in reversed(n["children"])
                    )
                else:
                    leaves.append(n)
            units = (
                [run["id"]]
                if run
                else list(dict.fromkeys(n["round_id"] or n["id"] for n in leaves))
            )
            result = []
            for unit in units[int(cursor) : int(cursor) + limit]:
                r = db.execute(
                    "SELECT * FROM runs WHERE id=? AND scope=?", (unit, scope)
                ).fetchone()
                matches = db.execute(
                    "SELECT id FROM nodes WHERE scope=? AND round_id=? ORDER BY lo,seq",
                    (scope, unit),
                ).fetchall()
                if matches:
                    result.append(
                        {
                            "id": unit,
                            "round": dict(r) if r else {"status": "legacy_inferred"},
                            "records": [self.node(scope, x[0], db) for x in matches],
                        }
                    )
                else:
                    result.append(
                        {
                            "id": unit,
                            "round": None,
                            "records": [self.node(scope, unit, db)],
                        }
                    )
            return {
                "units": result,
                "next_cursor": str(int(cursor) + limit)
                if len(units) > int(cursor) + limit
                else None,
            }

    def recover(self):
        with self.connection(True) as db:
            db.execute(
                "UPDATE nodes SET active=1 WHERE round_id IN (SELECT id FROM runs WHERE status='running')"
            )
            db.execute(
                "UPDATE runs SET status='interrupted',ended=?,error='process_restarted' WHERE status='running'",
                (time.time(),),
            )
            db.execute("UPDATE nodes SET reservation=NULL")
            db.execute("UPDATE jobs SET status='retry',retry=0 WHERE status='running'")
            db.execute(
                "UPDATE media SET status='missing',error='process_restarted_during_capture' WHERE status='pending'"
            )

    def plan_job(self, scope, budget):
        view = self.view(scope)
        with self.connection(True) as db:
            active = db.execute(
                "SELECT id FROM jobs WHERE scope=? AND status IN ('queued','running','ready','retry')",
                (scope,),
            ).fetchone()
            if active:
                return active[0]
            for level in sorted({n["level"] for n in view["nodes"]}):
                nodes = [n for n in view["nodes"] if n["level"] == level]
                total = sum(n["tokens"] for n in nodes)
                if total < 2 * budget:
                    continue
                candidates, size = [], 0
                for n in nodes:
                    if n["reservation"]:
                        break
                    if n["round_id"]:
                        run = db.execute(
                            "SELECT status FROM runs WHERE id=?", (n["round_id"],)
                        ).fetchone()
                        if run and run[0] == "running":
                            break
                    # Retain boundary units; a single giant old unit may advance whole.
                    if candidates and size + n["tokens"] > budget:
                        break
                    candidates.append(n)
                    size += n["tokens"]
                    if size >= budget:
                        break
                # A boundary may fall in an interleaved or multi-message round. Roll
                # it back to that round's first occurrence until the prefix is safe.
                while candidates:
                    cut = candidates[-1]["hi"]
                    split_start = None
                    for n in candidates:
                        if n["round_id"]:
                            span = db.execute(
                                "SELECT MIN(lo),MAX(hi) FROM nodes WHERE scope=? AND round_id=?",
                                (scope, n["round_id"]),
                            ).fetchone()
                            if span[1] > cut:
                                split_start = (
                                    span[0]
                                    if split_start is None
                                    else min(split_start, span[0])
                                )
                    if split_start is None:
                        break
                    candidates = [n for n in candidates if n["hi"] < split_start]
                # If the first round itself is larger than B, advance its complete
                # chronological envelope, provided B of newer content stays intact.
                if not candidates and nodes:
                    envelope = nodes[0]["hi"]
                    for n in nodes:
                        if n["lo"] > envelope:
                            break
                        if n["round_id"]:
                            span = db.execute(
                                "SELECT MAX(hi) FROM nodes WHERE scope=? AND round_id=?",
                                (scope, n["round_id"]),
                            ).fetchone()[0]
                            envelope = max(envelope, span)
                    proposed = [n for n in nodes if n["hi"] <= envelope]
                    if all(not n["reservation"] for n in proposed):
                        running = any(
                            db.execute(
                                "SELECT 1 FROM runs WHERE id=? AND status='running'",
                                (n["round_id"],),
                            ).fetchone()
                            for n in proposed
                        )
                        if not running:
                            candidates = proposed
                size = sum(n["tokens"] for n in candidates)
                if not candidates or total - size < budget:
                    continue
                sources = [n["id"] for n in candidates]
                # Read back inside the transaction to reject a simultaneous reserve.
                if any(self.node(scope, i, db)["reservation"] for i in sources):
                    continue
                ident = "j_" + uuid4().hex
                db.execute(
                    "INSERT INTO jobs(id,scope,level,sources,status,created) VALUES(?,?,?,?,'queued',?)",
                    (ident, scope, level, encode(sources), time.time()),
                )
                return ident
        return None

    def next_job(self):
        with self.connection(True) as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','retry','ready') AND retry<=? ORDER BY created LIMIT 1",
                (time.time(),),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            db.execute(
                "UPDATE jobs SET status=?,attempts=attempts+1 WHERE id=?",
                ("ready" if row["status"] == "ready" else "running", row["id"]),
            )
            result["sources"] = json.loads(result["sources"])
            return result

    def job_result(self, ident, text):
        with self.connection(True) as db:
            db.execute(
                "UPDATE jobs SET status='ready',result=? WHERE id=?", (text, ident)
            )

    def fail_job(self, ident, error):
        with self.connection(True) as db:
            attempts = db.execute(
                "SELECT attempts FROM jobs WHERE id=?", (ident,)
            ).fetchone()[0]
            db.execute(
                "UPDATE jobs SET status='retry',error=?,retry=? WHERE id=?",
                (error, time.time() + min(3600, 30 * 2 ** min(attempts, 7)), ident),
            )

    def publish(self, job):
        with self.connection(True) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if row["status"] == "complete":
                return True
            if row["status"] != "ready":
                raise ValueError("summary_not_ready")
            sources = [
                self.node(job["scope"], i, db) for i in json.loads(row["sources"])
            ]
            if any(not n["active"] or n["reservation"] for n in sources):
                db.execute(
                    "UPDATE jobs SET status='conflict',error='sources_changed',ended=? WHERE id=?",
                    (time.time(), job["id"]),
                )
                return False
            for n in sources:
                if (
                    n["round_id"]
                    and db.execute(
                        "SELECT 1 FROM runs WHERE id=? AND status='running'",
                        (n["round_id"],),
                    ).fetchone()
                ):
                    raise ValueError("source_round_running")
            text = row["result"]
            tokens = self.count({"role": "user", "content": text})
            if (
                not text
                or not text.strip()
                or tokens >= sum(n["tokens"] for n in sources)
            ):
                raise ValueError("summary_not_shorter_or_empty")
            ident = "s_" + uuid4().hex
            db.execute(
                """INSERT INTO nodes(id,scope,level,kind,payload,children,tokens,lo,hi,timestamp,received)
                VALUES(?,?,?,'summary',?,?,?,?,?,?,?)""",
                (
                    ident,
                    job["scope"],
                    row["level"] + 1,
                    encode({"text": text}),
                    row["sources"],
                    tokens,
                    min(n["lo"] for n in sources),
                    max(n["hi"] for n in sources),
                    sources[-1]["timestamp"],
                    time.time(),
                ),
            )
            for n in sources:
                db.execute("UPDATE nodes SET active=0 WHERE id=?", (n["id"],))
            db.execute(
                "UPDATE scopes SET version=version+1 WHERE id=?", (job["scope"],)
            )
            db.execute(
                "UPDATE jobs SET status='complete',error='',ended=? WHERE id=?",
                (time.time(), job["id"]),
            )
            return True
