"""Durable originals, summary provenance, and atomic working projections."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from .history_model import history_key, ordered_history_nodes, safe_history


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
                CREATE TABLE IF NOT EXISTS native_histories (
                    scope TEXT PRIMARY KEY, messages TEXT NOT NULL,
                    ids TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS retirements (
                    scope TEXT NOT NULL, node TEXT NOT NULL, replacement TEXT,
                    reason TEXT NOT NULL, created REAL NOT NULL,
                    PRIMARY KEY(scope,node)
                );
                CREATE TABLE IF NOT EXISTS pending_histories (
                    run TEXT PRIMARY KEY, scope TEXT NOT NULL,
                    payload TEXT NOT NULL, created REAL NOT NULL
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
            if "committed" not in columns:
                db.execute(
                    "ALTER TABLE runs ADD COLUMN committed INTEGER NOT NULL DEFAULT 1"
                )

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

    def scope_origin(self, scope):
        with self.connection() as db:
            row = db.execute(
                "SELECT origin FROM scopes WHERE id=?", (scope,)
            ).fetchone()
        if row is None:
            raise ValueError("session_not_found")
        return row[0]

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
        reservation=None,
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
                source_key,sender,timestamp,received,reservation) VALUES(?,?,0,'raw',?,?,?,?,?,?,?,?)""",
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
                    reservation,
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

    def native_history(self, scope):
        """Return the last successfully reconciled native history and its sidecar IDs."""
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM native_histories WHERE scope=?", (scope,)
            ).fetchone()
            if row is None:
                return None
            return {
                "messages": json.loads(row["messages"]),
                "ids": json.loads(row["ids"]),
                "revision": row["revision"],
            }

    def stage_history(self, run, scope, payload):
        """Persist reconciliation metadata before committing to the native DB."""
        with self.connection(True) as db:
            db.execute(
                """INSERT INTO pending_histories(run,scope,payload,created) VALUES(?,?,?,?)
                ON CONFLICT(run) DO UPDATE SET scope=excluded.scope,payload=excluded.payload,created=excluded.created""",
                (run, scope, encode(payload), time.time()),
            )

    def pending_histories(self, scope):
        """Read durable commit intents; model messages stay ordinary dictionaries."""
        with self.connection() as db:
            return [
                {
                    "run": row["run"],
                    "scope": row["scope"],
                    "payload": json.loads(row["payload"]),
                }
                for row in db.execute(
                    "SELECT * FROM pending_histories WHERE scope=? ORDER BY created",
                    (scope,),
                )
            ]

    def clear_history(self, run):
        """Clear an applied or explicitly failed native commit intent."""
        with self.connection(True) as db:
            db.execute("DELETE FROM pending_histories WHERE run=?", (run,))

    def sync_history(
        self,
        scope,
        before,
        after,
        *,
        candidates=(),
        run=None,
        internal=False,
        editable=(),
        excluded=(),
    ):
        """Follow successful native history edits without rewriting original records.

        Only messages represented in the native sidecar may be removed by a
        full-list write. Unpublished group messages and concurrent round tails
        remain independent. Immutable replacement nodes make source IDs exact
        versions for background summary conflict detection.
        """
        before, after = safe_history(before), safe_history(after)
        editable = set(editable)
        excluded = set(excluded)
        with self.connection(True) as db:
            row = db.execute(
                "SELECT * FROM native_histories WHERE scope=?", (scope,)
            ).fetchone()
            aliases = defaultdict(set)
            explicit_candidates = set()
            candidate_priority = {}
            projected_aliases = defaultdict(set)
            for priority, candidate in enumerate(candidates):
                ident = candidate.get("id")
                if ident and isinstance(candidate.get("message"), dict):
                    explicit_candidates.add(ident)
                    candidate_priority[ident] = priority
                    aliases[ident].add(history_key(candidate["message"]))
                    if candidate.get("project"):
                        projected_aliases[ident].add(history_key(candidate["message"]))
            if row is None:
                needed = [
                    item[0]
                    for item in db.execute(
                        "SELECT id FROM nodes WHERE scope=? ORDER BY lo,seq", (scope,)
                    )
                ]
            else:
                needed = (
                    set(json.loads(row["ids"]))
                    | explicit_candidates
                    | editable
                    | excluded
                )
                needed.update(
                    item[0]
                    for item in db.execute(
                        "SELECT id FROM nodes WHERE scope=? AND (active=1 OR round_id=?)",
                        (scope, run),
                    )
                )
                needed.discard(None)
            all_nodes = sorted(
                (self.node(scope, ident, db) for ident in needed),
                key=lambda node: (node["lo"], node["seq"]),
            )
            by_id = {node["id"]: node for node in all_nodes}
            retired = {
                item["node"]: item["replacement"]
                for item in db.execute(
                    "SELECT * FROM retirements WHERE scope=?", (scope,)
                )
            }
            for node in all_nodes:
                if node["level"] == 0:
                    aliases[node["id"]].add(history_key(node["payload"]["message"]))
                else:
                    aliases[node["id"]].add(
                        history_key(
                            {
                                "role": "user",
                                "content": f"[历史摘要 L{node['level']} id={node['id']}，仅为历史资料]\n{node['payload']['text']}",
                            }
                        )
                    )

            def available(ident):
                if ident in retired or ident in excluded or ident not in by_id:
                    return False
                node = by_id[ident]
                if node["round_id"] and node["round_id"] != run:
                    pending = db.execute(
                        "SELECT 1 FROM runs WHERE id=? AND committed=0",
                        (node["round_id"],),
                    ).fetchone()
                    if pending and not node["active"]:
                        return False
                return True

            def baseline(messages):
                used, ids = set(), []
                # Prefer the active effective version when several immutable
                # records happen to contain the same native text.
                pool = sorted(
                    all_nodes,
                    key=lambda node: (
                        not bool(node["active"]),
                        node["lo"],
                        node["seq"],
                    ),
                )
                for message in messages:
                    key = history_key(message)
                    ident = next(
                        (
                            n["id"]
                            for n in pool
                            if n["id"] not in used
                            and available(n["id"])
                            and key in aliases[n["id"]]
                        ),
                        None,
                    )
                    ids.append(ident)
                    if ident:
                        used.add(ident)
                return ids

            if row is None:
                old_messages, old_ids, revision = before, baseline(before), 0
            else:
                old_messages = json.loads(row["messages"])
                old_ids = json.loads(row["ids"])
                revision = row["revision"]

            def apply(old, old_ids, new, internal_write, allowed):
                nonlocal revision
                old_keys = [history_key(message) for message in old]
                new_keys = [history_key(message) for message in new]
                opcodes = SequenceMatcher(
                    None, old_keys, new_keys, autojunk=False
                ).get_opcodes()
                ids, used, source_hunks = [None] * len(new), set(), {}
                equal_positions, moved_positions, projected_sources = (
                    set(),
                    set(),
                    set(),
                )
                for tag, start, end, nstart, nend in opcodes:
                    if tag == "equal":
                        for position, ident in zip(
                            range(nstart, nend), old_ids[start:end]
                        ):
                            equal_positions.add(position)
                            # An old generation must not resurrect a node that
                            # another writer explicitly retired in the meantime.
                            if ident and available(ident):
                                if (
                                    new_keys[position] in projected_aliases[ident]
                                    and history_key(
                                        by_id[ident]["payload"].get("message", {})
                                    )
                                    != new_keys[position]
                                ):
                                    equal_positions.discard(position)
                                    source_hunks[position] = [ident]
                                    projected_sources.add(ident)
                                else:
                                    ids[position] = ident
                                    used.add(ident)
                    else:
                        sources = [ident for ident in old_ids[start:end] if ident]
                        for position in range(nstart, nend):
                            # Equal-size edits keep positional provenance. Giving
                            # every new message the whole hunk would map a fresh
                            # input to an unrelated earlier replacement.
                            if end - start == nend - nstart:
                                ident = old_ids[start + position - nstart]
                                source_hunks[position] = [ident] if ident else []
                            else:
                                source_hunks[position] = sources

                for position, key in enumerate(new_keys):
                    if ids[position] is not None or position in equal_positions:
                        continue
                    # Existing raw output aliases are matched before a visible
                    # revision is created. They may still await native commit.
                    choices = [
                        ident
                        for ident in aliases
                        if ident not in used
                        and available(ident)
                        and key in aliases[ident]
                        and (
                            ident in explicit_candidates
                            or ident in old_ids
                            or run
                            and by_id[ident]["round_id"] == run
                        )
                    ]
                    choices.sort(
                        key=lambda ident: (
                            by_id[ident]["round_id"] != run if run else False,
                            -candidate_priority.get(ident, -1),
                            not bool(by_id[ident]["active"]),
                            by_id[ident]["lo"],
                            by_id[ident]["seq"],
                        )
                    )
                    if choices:
                        ident = choices[0]
                        if (
                            key in projected_aliases[ident]
                            and history_key(by_id[ident]["payload"].get("message", {}))
                            != key
                        ):
                            source_hunks[position] = [ident]
                            projected_sources.add(ident)
                        elif ident in old_ids:
                            # Reordering is a new effective version too. Keep
                            # archive chronology intact and position its new
                            # version between the retained native anchors.
                            source_hunks[position] = [ident]
                            moved_positions.add(position)
                        else:
                            ids[position] = ident
                            used.add(ident)

                removed = (
                    excluded
                    | projected_sources
                    | {
                        ident
                        for ident in old_ids
                        if ident
                        and ident not in used
                        and (not internal_write or ident in allowed)
                    }
                )
                changed = set()
                for position, message in enumerate(new):
                    if ids[position] is not None or position in equal_positions:
                        continue
                    sources = source_hunks.get(position, [])
                    if internal_write:
                        # A stale base replacement has no authority over an
                        # intervening external edit or another round's records.
                        if sources and any(
                            ident in retired or ident not in allowed
                            for ident in sources
                        ):
                            continue
                        if not sources and key_matches_retired(new_keys[position]):
                            continue
                    ident = "v_" + uuid4().hex
                    source_nodes = [by_id[source] for source in sources]
                    left = next(
                        (by_id[ids[i]] for i in range(position - 1, -1, -1) if ids[i]),
                        None,
                    )
                    right = next(
                        (
                            by_id[ids[i]]
                            for i in range(position + 1, len(ids))
                            if ids[i]
                        ),
                        None,
                    )
                    if source_nodes and position not in moved_positions:
                        lo, hi = (
                            min(n["lo"] for n in source_nodes),
                            max(n["hi"] for n in source_nodes),
                        )
                    elif left and right and right["lo"] > left["hi"]:
                        lo = hi = (left["hi"] + right["lo"]) / 2
                    elif left:
                        lo = hi = left["hi"] + 0.5
                    elif right:
                        lo = hi = right["lo"] - 0.5
                    else:
                        lo = hi = db.execute(
                            "SELECT COALESCE(MAX(hi),0)+1 FROM nodes WHERE scope=?",
                            (scope,),
                        ).fetchone()[0]
                    source_rounds = {node["round_id"] for node in source_nodes}
                    round_id = run or (
                        next(iter(source_rounds)) if len(source_rounds) == 1 else None
                    )
                    stamp = (
                        source_nodes[0]["timestamp"] if source_nodes else time.time()
                    )
                    extra = {"native_history_revision": True}
                    if source_nodes and all(
                        source["payload"].get("extra", {}).get("time_unknown")
                        for source in source_nodes
                    ):
                        extra["time_unknown"] = True
                    cur = db.execute(
                        """INSERT INTO nodes(id,scope,level,kind,payload,children,tokens,lo,hi,
                        round_id,sender,timestamp,received) VALUES(?,?,0,'revision',?,?,?,?,?,?,?,?,?)""",
                        (
                            ident,
                            scope,
                            encode(
                                {
                                    "message": message,
                                    "extra": extra,
                                }
                            ),
                            encode(sources),
                            self.count(message),
                            lo,
                            hi,
                            round_id,
                            source_nodes[0]["sender"] if source_nodes else "",
                            stamp,
                            time.time(),
                        ),
                    )
                    node = self.node(scope, ident, db)
                    node["seq"] = cur.lastrowid
                    by_id[ident] = node
                    aliases[ident].add(new_keys[position])
                    ids[position] = ident
                    used.add(ident)
                    changed.add(ident)

                # Retire first, then restore valid descendants. This prevents a
                # multi-message edit from briefly reviving another deleted source.
                replacements = {}
                for ident in removed:
                    replacement = next(
                        (
                            ids[position]
                            for position, sources in source_hunks.items()
                            if ident in sources
                            and ids[position]
                            and ids[position] not in old_ids
                        ),
                        None,
                    )
                    replacements[ident] = replacement
                for ident, replacement in replacements.items():
                    if ident not in retired:
                        retired[ident] = replacement
                        db.execute(
                            "INSERT INTO retirements VALUES(?,?,?,?,?)",
                            (
                                scope,
                                ident,
                                replacement,
                                "native_history_edit",
                                time.time(),
                            ),
                        )
                        db.execute(
                            "UPDATE nodes SET active=0 WHERE scope=? AND id=?",
                            (scope, ident),
                        )
                        changed.add(ident)
                if removed:
                    self._invalidate_summaries(db, scope, removed, retired)
                for ident in used:
                    if available(ident):
                        # Raw sources already represented by a valid active
                        # summary stay compressed unless that summary was edited.
                        covered = self._covered_by_active_summary(db, scope, ident)
                        if not covered:
                            db.execute(
                                "UPDATE nodes SET active=1 WHERE scope=? AND id=?",
                                (scope, ident),
                            )
                revision += int(old_keys != new_keys or old_ids != ids or bool(changed))
                return ids

            def key_matches_retired(key):
                return any(key in aliases[ident] for ident in retired)

            # Repair a successful native write that was interrupted before the
            # sidecar commit, using only previously published IDs as authority.
            if [history_key(message) for message in old_messages] != [
                history_key(message) for message in before
            ]:
                old_ids = apply(old_messages, old_ids, before, False, set())
                old_messages = before
            ids = apply(old_messages, old_ids, after, internal, editable)
            if run:
                own = [
                    item["id"]
                    for item in db.execute(
                        "SELECT id,kind,source_key FROM nodes WHERE scope=? AND round_id=? AND level=0",
                        (scope, run),
                    )
                    if item["kind"] == "revision"
                    or (item["source_key"] or "").startswith(run + ":")
                ]
                hidden = (set(own) | excluded) - {ident for ident in ids if ident}
                for ident in hidden:
                    if ident not in retired:
                        retired[ident] = None
                        db.execute(
                            "INSERT INTO retirements VALUES(?,?,?,?,?)",
                            (scope, ident, None, "native_unsaved", time.time()),
                        )
                        db.execute(
                            "UPDATE nodes SET active=0 WHERE scope=? AND id=?",
                            (scope, ident),
                        )
                if hidden:
                    self._invalidate_summaries(db, scope, hidden, retired)
                db.execute(
                    "UPDATE runs SET committed=1 WHERE scope=? AND id=?", (scope, run)
                )
            elif excluded:
                for ident in excluded:
                    if ident not in retired:
                        retired[ident] = None
                        db.execute(
                            "INSERT INTO retirements VALUES(?,?,?,?,?)",
                            (scope, ident, None, "native_no_save", time.time()),
                        )
                        db.execute(
                            "UPDATE nodes SET active=0 WHERE scope=? AND id=?",
                            (scope, ident),
                        )
                self._invalidate_summaries(db, scope, excluded, retired)
            db.execute(
                """INSERT INTO native_histories(scope,messages,ids,revision) VALUES(?,?,?,?)
                ON CONFLICT(scope) DO UPDATE SET messages=excluded.messages,ids=excluded.ids,revision=excluded.revision""",
                (scope, encode(after), encode(ids), revision),
            )
            db.execute("UPDATE scopes SET version=version+1 WHERE id=?", (scope,))
            return {"messages": after, "ids": ids, "revision": revision}

    def _covered_by_active_summary(self, db, scope, ident):
        summaries = {
            row[0]: json.loads(row[1])
            for row in db.execute(
                "SELECT id,children FROM nodes WHERE scope=? AND kind='summary'",
                (scope,),
            )
        }
        pending = [
            row[0]
            for row in db.execute(
                "SELECT id FROM nodes WHERE scope=? AND active=1 AND kind='summary'",
                (scope,),
            )
        ]
        seen = set()
        while pending:
            parent = pending.pop()
            if parent in seen:
                continue
            seen.add(parent)
            children = summaries.get(parent, [])
            if ident in children:
                return True
            pending.extend(child for child in children if child in summaries)
        return False

    def _invalidate_summaries(self, db, scope, sources, retired):
        summaries = {
            row["id"]: {**dict(row), "children": json.loads(row["children"])}
            for row in db.execute(
                "SELECT id,children,active FROM nodes WHERE scope=? AND kind='summary'",
                (scope,),
            )
        }
        affected = set(sources)
        changed = True
        while changed:
            changed = False
            for ident, node in summaries.items():
                if ident not in affected and affected.intersection(node["children"]):
                    affected.add(ident)
                    changed = True
        invalid = affected.intersection(summaries)
        roots = [
            summaries[ident]
            for ident in invalid
            if summaries[ident]["active"] and ident not in sources
        ]
        for ident in invalid:
            db.execute(
                "UPDATE nodes SET active=0 WHERE scope=? AND id=?", (scope, ident)
            )
            if ident not in retired:
                retired[ident] = None
                db.execute(
                    "INSERT INTO retirements VALUES(?,?,?,?,?)",
                    (scope, ident, None, "summary_source_edited", time.time()),
                )

        def restore(ident, visited):
            if ident in visited:
                return
            visited.add(ident)
            if ident in retired:
                if retired[ident]:
                    restore(retired[ident], visited)
                elif ident in invalid:
                    for child in summaries[ident]["children"]:
                        restore(child, visited)
                return
            node = db.execute(
                "SELECT round_id,active FROM nodes WHERE scope=? AND id=?",
                (scope, ident),
            ).fetchone()
            pending = (
                db.execute(
                    "SELECT 1 FROM runs WHERE id=? AND committed=0", (node["round_id"],)
                ).fetchone()
                if node["round_id"]
                else None
            )
            if pending and not node["active"]:
                return
            if not self._covered_by_active_summary(db, scope, ident):
                db.execute(
                    "UPDATE nodes SET active=1 WHERE scope=? AND id=?", (scope, ident)
                )

        for root in roots:
            for child in root["children"]:
                restore(child, set())

        for job in db.execute(
            "SELECT id,sources FROM jobs WHERE scope=? AND status IN ('queued','running','ready','retry')",
            (scope,),
        ).fetchall():
            if affected.intersection(json.loads(job["sources"])):
                db.execute(
                    "UPDATE jobs SET status='conflict',error='sources_changed',ended=? WHERE id=?",
                    (time.time(), job["id"]),
                )

    def history_nodes(self, nodes):
        """Assemble valid tool blocks without changing archive/run ownership."""
        nodes = list(nodes)
        scopes = {node["scope"] for node in nodes if node.get("round_id")}
        inactive = {}
        if scopes:
            with self.connection() as db:
                inactive = dict(
                    db.execute(
                        "SELECT id,status FROM runs WHERE status IN "
                        "('failed','cancelled','interrupted') AND scope IN ("
                        + ",".join("?" for _ in scopes)
                        + ")",
                        tuple(scopes),
                    )
                )
        return ordered_history_nodes(nodes, inactive)

    def view(self, scope, exclude=(), *, include_pending=True):
        with self.connection() as db:
            state = db.execute("SELECT * FROM scopes WHERE id=?", (scope,)).fetchone()
            if state is None:
                raise ValueError("session_not_found")
            nodes = [
                self.node(scope, r[0], db)
                for r in db.execute(
                    "SELECT id FROM nodes WHERE scope=? AND active=1 "
                    "AND (? OR reservation IS NULL OR reservation NOT LIKE 'incoming:%') "
                    "ORDER BY lo,seq",
                    (scope, include_pending),
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

    def reserve(self, scope, ids, owner, *, previous_owners=()):
        """Reserve current input versions while keeping their original IDs traceable."""
        resolved = []
        allowed = {None, owner, *previous_owners}
        with self.connection(True) as db:
            for ident in ids:
                visited = set()
                while ident not in visited:
                    visited.add(ident)
                    row = db.execute(
                        "SELECT active,reservation FROM nodes WHERE scope=? AND id=?",
                        (scope, ident),
                    ).fetchone()
                    if row is None:
                        break
                    if row[0]:
                        if row[1] not in allowed:
                            break
                        if ident not in resolved:
                            resolved.append(ident)
                        break
                    replacement = db.execute(
                        "SELECT replacement FROM retirements WHERE scope=? AND node=?",
                        (scope, ident),
                    ).fetchone()
                    if not replacement or not replacement[0]:
                        break
                    ident = replacement[0]
                else:
                    row = None
                if row is None or not row[0] or row[1] not in allowed:
                    raise ValueError("input_already_compacted_or_reserved")
                db.execute(
                    "UPDATE nodes SET reservation=? WHERE scope=? AND id=?",
                    (owner, scope, ident),
                )
        return resolved

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
                "INSERT INTO runs(id,scope,version,base,status,started,committed) VALUES(?,?,?,?,'running',?,0)",
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

    def finish(self, run, status="complete", error="", activate=True):
        with self.connection(True) as db:
            scope = db.execute(
                "SELECT scope FROM runs WHERE id=? AND status='running'", (run,)
            ).fetchone()
            if scope:
                db.execute(
                    "UPDATE runs SET status=?,ended=?,error=?,committed=? WHERE id=?",
                    (status, time.time(), error, int(activate), run),
                )
                if activate:
                    db.execute(
                        """UPDATE nodes SET active=1 WHERE round_id=? AND id NOT IN
                        (SELECT node FROM retirements WHERE scope=?)""",
                        (run, scope[0]),
                    )
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
        projected=None,
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
                n = (
                    projected.get(row[0])
                    if projected is not None
                    else self.node(scope, row[0], db)
                )
                if n is None:
                    continue
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
            if (
                target == "children"
                and root
                and (root["level"] > 0 or root["kind"] == "revision")
            ):
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
                if n["level"] or n["kind"] == "revision" and n["children"]:
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
                    "SELECT id FROM nodes WHERE scope=? AND round_id=? AND kind='raw' ORDER BY lo,seq",
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
                "UPDATE nodes SET active=1 WHERE round_id IN (SELECT id FROM runs WHERE status='running' AND committed=1) AND id NOT IN (SELECT node FROM retirements)"
            )
            db.execute(
                "UPDATE runs SET status='interrupted',ended=?,error='process_restarted' WHERE status='running'",
                (time.time(),),
            )
            # Keep unfinished output archival only. Native reconciliation can
            # still recognize a write that succeeded immediately before exit.
            db.execute("UPDATE runs SET committed=1 WHERE committed=0")
            db.execute("UPDATE nodes SET reservation=NULL")
            db.execute("UPDATE jobs SET status='retry',retry=0 WHERE status='running'")
            db.execute(
                "UPDATE media SET status='missing',error='process_restarted_during_capture' WHERE status='pending'"
            )

    def plan_job(self, scope, budget, *, nodes=None):
        with self.connection(True) as db:
            view = {
                "nodes": sorted(nodes, key=lambda n: (n["lo"], n["seq"]))
                if nodes is not None
                else [
                    self.node(scope, item[0], db)
                    for item in db.execute(
                        "SELECT id FROM nodes WHERE scope=? AND active=1 ORDER BY lo,seq",
                        (scope,),
                    )
                ]
            }
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
                            "SELECT status,committed FROM runs WHERE id=?",
                            (n["round_id"],),
                        ).fetchone()
                        if run and (run[0] == "running" or not run[1]):
                            break
                    # Retain boundary units; a single giant old unit may advance whole.
                    if size > 0 and size + n["tokens"] > budget:
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
                                "SELECT 1 FROM runs WHERE id=? AND (status='running' OR committed=0)",
                                (n["round_id"],),
                            ).fetchone()
                            for n in proposed
                        )
                        if not running:
                            candidates = proposed
                size = sum(n["tokens"] for n in candidates)
                if not candidates or size <= 0 or total - size < budget:
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
                "UPDATE jobs SET status='ready',result=? WHERE id=? AND status IN ('queued','running','retry','ready')",
                (text, ident),
            )

    def fail_job(self, ident, error):
        with self.connection(True) as db:
            row = db.execute(
                "SELECT attempts,status FROM jobs WHERE id=?", (ident,)
            ).fetchone()
            if row["status"] in ("conflict", "complete"):
                return
            attempts = row["attempts"]
            db.execute(
                "UPDATE jobs SET status='retry',error=?,retry=? WHERE id=?",
                (error, time.time() + min(3600, 30 * 2 ** min(attempts, 7)), ident),
            )

    def publish(self, job):
        with self.connection(True) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if row["status"] == "complete":
                return True
            if row["status"] == "conflict":
                return False
            if row["status"] != "ready":
                raise ValueError("summary_not_ready")
            sources = [
                self.node(job["scope"], i, db) for i in json.loads(row["sources"])
            ]
            if any(
                not n["active"]
                or n["reservation"]
                or db.execute(
                    "SELECT 1 FROM retirements WHERE scope=? AND node=?",
                    (job["scope"], n["id"]),
                ).fetchone()
                for n in sources
            ):
                db.execute(
                    "UPDATE jobs SET status='conflict',error='sources_changed',ended=? WHERE id=?",
                    (time.time(), job["id"]),
                )
                return False
            for n in sources:
                if (
                    n["round_id"]
                    and db.execute(
                        "SELECT 1 FROM runs WHERE id=? AND (status='running' OR committed=0)",
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
