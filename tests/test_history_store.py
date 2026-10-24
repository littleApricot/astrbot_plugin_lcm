"""Exercise authoritative history edits independently of models and AstrBot."""

import copy
import time
from uuid import uuid4

from astrbot_plugin_lcm.history_model import history_key, safe_history
from astrbot_plugin_lcm.store import Store, encode


def message(text, role="user", **extra):
    return {"role": role, "content": text, **extra}


def make_store(tmp_path):
    store = Store(
        tmp_path / "history.sqlite3",
        count=lambda value: max(1, len(encode(value)) // 8),
    )
    return store, store.scope("group", "conversation")


def seed(store, scope, messages, tokens=20):
    ids = [store.append(scope, item, tokens=tokens) for item in messages]
    store.sync_history(scope, messages, messages)
    return ids


def active_ids(store, scope):
    return [item["id"] for item in store.view(scope)["nodes"]]


def add_summary(store, scope, children, level):
    ident = "s_" + uuid4().hex
    nodes = [store.node(scope, child) for child in children]
    with store.connection(True) as db:
        db.execute(
            """INSERT INTO nodes(id,scope,level,kind,payload,children,tokens,lo,hi,timestamp,received)
            VALUES(?,?,?,'summary',?,?,2,?,?,?,?)""",
            (
                ident,
                scope,
                level,
                encode({"text": "history summary"}),
                encode(children),
                min(node["lo"] for node in nodes),
                max(node["hi"] for node in nodes),
                time.time(),
                time.time(),
            ),
        )
        for child in children:
            db.execute("UPDATE nodes SET active=0 WHERE id=?", (child,))
    return ident


def test_safe_history_pairs_are_partial_and_original_is_untouched():
    calls = [
        {"id": name, "type": "function", "function": {"name": name, "arguments": "{}"}}
        for name in ("visible", "hidden")
    ]
    original = [
        message("hello"),
        {"role": "_checkpoint", "content": {"id": "checkpoint"}},
        message("keep this text", "assistant", tool_calls=calls),
        message("visible result", "tool", tool_call_id="visible"),
        message("private result", "tool", tool_call_id="hidden", _no_save=True),
        message("orphan", "tool", tool_call_id="unknown"),
        message(None, "assistant", tool_calls=[{"id": "missing"}]),
        message(
            [
                {"type": "text", "text": "kept"},
                {"type": "text", "text": "private", "_no_save": True},
            ]
        ),
    ]
    pristine = copy.deepcopy(original)
    safe = safe_history(original)
    assert original == pristine
    assert len(safe) == 4
    assert safe[1]["content"] == "keep this text"
    assert [call["id"] for call in safe[1]["tool_calls"]] == ["visible"]
    assert safe[-1]["content"] == [{"type": "text", "text": "kept"}]
    assert history_key(message("ab")) == history_key(
        message(
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            _no_save=False,
            tool_calls=None,
        )
    )


def test_external_change_delete_insert_keeps_original_and_silent_records(tmp_path):
    store, scope = make_store(tmp_path)
    before = [message("question"), message("old answer", "assistant")]
    ids = seed(store, scope, before)
    silent = store.append(scope, message("unanswered group message"))
    after = [
        before[0],
        message("corrected answer", "assistant"),
        message("a new history record"),
    ]
    result = store.sync_history(scope, before, after)
    assert result["ids"][0] == ids[0]
    assert result["revision"] == 1
    assert store.node(scope, ids[1])["payload"]["message"] == before[1]
    current = active_ids(store, scope)
    assert ids[1] not in current and silent in current
    assert all(ident in current for ident in result["ids"])
    revised = store.node(scope, result["ids"][1])
    assert revised["kind"] == "revision" and revised["children"] == [ids[1]]
    original = store.expand(scope, revised["id"])["units"][0]["records"][0]
    assert original["id"] == ids[1] and original["payload"]["message"] == before[1]
    store.sync_history(scope, after, [])
    assert active_ids(store, scope) == [silent]
    assert store.native_history(scope)["messages"] == []


def test_first_sync_does_not_infer_unknown_archive_deletions(tmp_path):
    store, scope = make_store(tmp_path)
    silent = store.append(scope, message("never published"))
    a = store.append(scope, message("published"))
    store.sync_history(scope, [message("published")], [])
    assert a not in active_ids(store, scope)
    assert silent in active_ids(store, scope)


def test_l2_ancestors_invalidate_and_unaffected_summary_restores(tmp_path):
    store, scope = make_store(tmp_path)
    before = [message(str(index)) for index in range(5)]
    ids = seed(store, scope, before)
    first = add_summary(store, scope, ids[:2], 1)
    second = add_summary(store, scope, ids[2:4], 1)
    top = add_summary(store, scope, [first, second], 2)
    after = [message("corrected old fact")] + before[1:]
    result = store.sync_history(scope, before, after)
    current = active_ids(store, scope)
    assert top not in current and first not in current and ids[0] not in current
    assert current == [result["ids"][0], ids[1], second, ids[4]]
    assert store.node(scope, ids[0])["payload"]["message"] == before[0]
    assert (
        store.expand(scope, result["ids"][0])["units"][0]["records"][0]["id"] == ids[0]
    )


def test_removing_a_published_summary_does_not_reactivate_its_children(tmp_path):
    store, scope = make_store(tmp_path)
    ids = [store.append(scope, message(str(index))) for index in range(3)]
    summary = add_summary(store, scope, ids[:2], 1)
    rendered = message(f"[历史摘要 L1 id={summary}，仅为历史资料]\nhistory summary")
    before = [rendered, message("2")]
    store.sync_history(scope, before, before)
    store.sync_history(scope, before, [before[-1]])
    assert active_ids(store, scope) == [ids[2]]
    assert store.node(scope, summary)["children"] == ids[:2]


def test_running_summary_is_invalidated_but_tail_append_is_not(tmp_path):
    store, scope = make_store(tmp_path)
    before = [message(str(index)) for index in range(4)]
    seed(store, scope, before)
    job_id = store.plan_job(scope, 40)
    job = store.next_job()
    tail = store.append(scope, message("silent new tail"))
    after = [message("edited source")] + before[1:]
    store.sync_history(scope, before, after)
    store.job_result(job_id, "obsolete summary")
    assert store.publish(job) is False
    store.fail_job(job_id, "must not retry an obsolete job")
    with store.connection() as db:
        assert (
            db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            == "conflict"
        )
    assert tail in active_ids(store, scope)


def test_duplicate_messages_and_text_part_aliases_do_not_duplicate_archive(tmp_path):
    store, scope = make_store(tmp_path)
    before = [message("same"), message("same"), message("tail")]
    ids = seed(store, scope, before)
    result = store.sync_history(
        scope, before, [message([{"type": "text", "text": "same"}]), before[2]]
    )
    assert result["ids"][0] in ids[:2] and result["ids"][1] == ids[2]
    assert active_ids(store, scope) == result["ids"]
    assert store.node(scope, ids[1])["payload"]["message"] == before[1]


def test_internal_stale_snapshot_does_not_revive_external_edit_or_drop_parallel_tail(
    tmp_path,
):
    store, scope = make_store(tmp_path)
    before = [message("question"), message("old answer", "assistant")]
    ids = seed(store, scope, before)
    edited = [before[0], message("external correction", "assistant")]
    external = store.sync_history(scope, before, edited)
    concurrent = store.append(scope, message("parallel tail", "assistant"))
    visible = edited + [message("parallel tail", "assistant")]
    store.sync_history(
        scope, edited, visible, candidates=[{"id": concurrent, "message": visible[-1]}]
    )
    stale = before + [
        message("my next question"),
        message("my own final answer", "assistant"),
    ]
    own_ids = [store.append(scope, item) for item in stale[2:]]
    result = store.sync_history(
        scope,
        visible,
        stale,
        internal=True,
        editable=ids + own_ids,
        candidates=[
            {"id": ident, "message": item} for ident, item in zip(own_ids, stale[2:])
        ],
    )
    current = active_ids(store, scope)
    assert ids[1] not in current
    assert external["ids"][1] in current and concurrent in current
    assert result["ids"][1] is None
    assert all(ident in current for ident in result["ids"][2:])


def test_native_commit_selects_saved_outputs_and_preserves_full_original_tool_chain(
    tmp_path,
):
    store, scope = make_store(tmp_path)
    current = message("choose a meme")
    inp = store.append(scope, current, tokens=25)
    store.reserve(scope, [inp], "owner")
    run = store.begin(scope, "owner", store.view(scope, exclude=[inp]), [inp])
    call = message(
        None,
        "assistant",
        tool_calls=[
            {
                "id": "search",
                "type": "function",
                "function": {"name": "find", "arguments": "{}"},
            }
        ],
    )
    result = message("tool result", "tool", tool_call_id="search")
    final = message(
        [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/meme.png"},
            }
        ],
        "assistant",
    )
    output_ids = [
        store.append(scope, item, round_id=run, source=f"{run}:{index}", tokens=25)
        for index, item in enumerate((call, result, final))
    ]
    store.finish(run, activate=False)
    store.append(scope, message("fresh tail"), tokens=50)
    assert store.plan_job(scope, 20) is None
    saved = store.sync_history(
        scope, [], [current, final], run=run, internal=True, editable=[inp] + output_ids
    )
    visible = active_ids(store, scope)
    assert saved["ids"] == [inp, output_ids[-1]]
    assert all(ident not in visible for ident in output_ids[:2])
    assert store.node(scope, output_ids[0])["payload"]["message"] == call
    assert store.node(scope, output_ids[1])["payload"]["message"] == result
    assert store.runs(scope)[0]["committed"] == 1


def test_reconcile_before_repairs_interrupted_sidecar_commit(tmp_path):
    store, scope = make_store(tmp_path)
    before = [message("old")]
    ids = seed(store, scope, before)
    actual_native = [message("changed during interrupted write")]
    repaired = store.sync_history(scope, actual_native, actual_native)
    assert repaired["ids"][0] != ids[0]
    assert ids[0] not in active_ids(store, scope)
    assert (
        store.node(scope, repaired["ids"][0])["payload"]["message"] == actual_native[0]
    )


def test_explicit_reorder_keeps_archive_order_and_moves_only_effective_version(
    tmp_path,
):
    store, scope = make_store(tmp_path)
    before = [message("first"), message("second"), message("third")]
    ids = seed(store, scope, before)
    after = [before[1], before[0], before[2]]
    result = store.sync_history(scope, before, after)
    assert [node["payload"]["message"] for node in store.view(scope)["nodes"]] == after
    assert store.node(scope, ids[0])["lo"] < store.node(scope, ids[1])["lo"]
    assert result["ids"][1] != ids[0] or result["ids"][0] != ids[1]


def test_new_identical_text_does_not_reuse_a_compressed_original(tmp_path):
    store, scope = make_store(tmp_path)
    original = store.append(scope, message("same text"))
    summary = add_summary(store, scope, [original], 1)
    rendered = message(f"[历史摘要 L1 id={summary}，仅为历史资料]\nhistory summary")
    store.sync_history(scope, [rendered], [rendered])
    after = [rendered, message("same text")]
    result = store.sync_history(scope, [rendered], after)
    assert result["ids"][1] != original
    assert result["ids"][1] in active_ids(store, scope)
    assert original not in active_ids(store, scope)


def test_native_cleanup_without_a_saved_history_keeps_unpublished_input(tmp_path):
    store, scope = make_store(tmp_path)
    inp = store.append(scope, message("must remain visible"))
    store.reserve(scope, [inp], "owner")
    run = store.begin(scope, "owner", store.view(scope, exclude=[inp]), [inp])
    temporary = store.append(
        scope,
        message("temporary tool state", "assistant"),
        round_id=run,
        source=f"{run}:0",
    )
    store.finish(run, "failed", activate=False)
    store.sync_history(scope, [], [], run=run, internal=True, editable=[inp, temporary])
    assert inp in active_ids(store, scope)
    assert temporary not in active_ids(store, scope)


def test_fresh_input_alias_wins_over_identical_unpublished_silent_alias(tmp_path):
    store, scope = make_store(tmp_path)
    old = store.append(scope, message("[older sender] hello"))
    fresh = store.append(scope, message("[new sender] hello"))
    native = message("hello")
    result = store.sync_history(
        scope,
        [],
        [native],
        candidates=[{"id": old, "message": native}, {"id": fresh, "message": native}],
    )
    assert result["ids"] == [fresh]
    store.sync_history(scope, [native], [])
    assert old in active_ids(store, scope) and fresh not in active_ids(store, scope)


def test_explicit_no_save_input_is_hidden_but_other_unpublished_inputs_remain(tmp_path):
    store, scope = make_store(tmp_path)
    hidden = store.append(scope, message("explicitly private"))
    silent = store.append(scope, message("unanswered remains"))
    store.sync_history(scope, [], [], excluded=[hidden])
    assert active_ids(store, scope) == [silent]
    assert (
        store.node(scope, hidden)["payload"]["message"]["content"]
        == "explicitly private"
    )


def test_projected_saved_alias_creates_revision_and_cannot_restore_private_parts(
    tmp_path,
):
    store, scope = make_store(tmp_path)
    raw = message(
        [
            {"type": "text", "text": "public"},
            {"type": "text", "text": "PRIVATE TOOL TRACE"},
        ],
        "assistant",
    )
    ident = store.append(scope, raw)
    saved = message([{"type": "text", "text": "public"}], "assistant")
    result = store.sync_history(
        scope,
        [],
        [saved],
        candidates=[{"id": ident, "message": saved, "project": True}],
    )
    current = store.view(scope)["nodes"]
    assert len(current) == 1 and current[0]["id"] == result["ids"][0] != ident
    assert current[0]["payload"]["message"] == saved
    assert current[0]["children"] == [ident]
    assert (
        store.expand(scope, current[0]["id"], target="children")["records"][0]["id"]
        == ident
    )
    assert store.node(scope, ident)["payload"]["message"] == raw
    assert (
        store.expand(scope, current[0]["id"])["units"][0]["records"][0]["id"] == ident
    )


def test_pending_native_intent_is_durable_scoped_and_independent_of_recover(tmp_path):
    store, scope = make_store(tmp_path)
    other = store.scope("other group", "conversation")
    payload = {
        "before": [],
        "after": [message("saved output", "assistant")],
        "candidates": [
            {
                "id": "raw",
                "message": message("saved output", "assistant"),
                "project": True,
            }
        ],
        "editable": ["raw"],
        "excluded": ["private input"],
    }
    original = copy.deepcopy(payload)
    store.stage_history("run", scope, payload)
    payload["after"].clear()
    restarted = Store(store.path)
    restarted.recover()
    assert restarted.pending_histories(scope) == [
        {"run": "run", "scope": scope, "payload": original}
    ]
    assert restarted.pending_histories(other) == []
    restarted.clear_history("run")
    assert restarted.pending_histories(scope) == []


def test_pending_native_intent_retry_replaces_only_its_own_run(tmp_path):
    store, scope = make_store(tmp_path)
    store.stage_history("first", scope, {"attempt": 1})
    store.stage_history("second", scope, {"attempt": 1})
    store.stage_history("first", scope, {"attempt": 2})
    assert {
        item["run"]: item["payload"] for item in store.pending_histories(scope)
    } == {"first": {"attempt": 2}, "second": {"attempt": 1}}
    store.clear_history("first")
    assert [item["run"] for item in store.pending_histories(scope)] == ["second"]
