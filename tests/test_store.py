"""Verify the user's boundaries and durability requirements with small budgets."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from astrbot_plugin_lcm.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(
        tmp_path / "lcm.sqlite3",
        count=lambda v: max(1, len(json.dumps(v, ensure_ascii=False)) // 10),
    )


def add(store, scope, tokens, text="history", **kw):
    return store.append(scope, {"role": "user", "content": text}, tokens=tokens, **kw)


def summarize(store, job_id, text="摘要"):
    job = store.next_job()
    assert job["id"] == job_id
    store.job_result(job_id, text)
    assert store.publish(job)
    return job


def test_boundary_35_plus_47_and_originals_survive(store):
    scope = store.scope("g", "c")
    ids = [add(store, scope, n) for n in (20, 15, 12, 35)]
    job = summarize(store, store.plan_job(scope, 40))
    assert job["sources"] == ids[:2]
    view = store.view(scope)
    assert sum(n["tokens"] for n in view["nodes"] if not n["level"]) == 47
    assert [n["id"] for n in view["nodes"] if n["level"] == 0] == ids[2:]
    assert len(store.history(scope)["records"]) == 4
    parent = view["nodes"][0]
    assert parent["children"] == ids[:2]
    assert len(store.expand(scope, parent["id"], limit=10)["units"]) == 2


def test_hidden_prefix_advances_with_first_effective_unit(store):
    scope = store.scope("g", "c")
    hidden = [add(store, scope, 1, "injected definition") for _ in range(2)]
    old = add(store, scope, 50, "old real chat")
    recent = add(store, scope, 50, "recent real chat")
    nodes = store.view(scope)["nodes"]
    for node in nodes:
        if node["id"] in hidden:
            node["tokens"] = 0
    ident = store.plan_job(scope, 40, nodes=nodes)
    job = store.next_job()
    assert ident and job["sources"] == hidden + [old]
    assert recent not in job["sources"]
    assert sum(node["tokens"] for node in nodes if node["id"] in job["sources"]) > 0


def test_whole_round_over_budget_and_interleaved_messages(store):
    scope = store.scope("g", "c")
    first = add(store, scope, 20)
    base = store.view(scope, exclude=[first])
    store.reserve(scope, [first], "owner")
    run = store.begin(scope, "owner", base, [first])
    silent = add(store, scope, 5, "silent")
    outputs = [add(store, scope, 30, round_id=run), add(store, scope, 30, round_id=run)]
    add(store, scope, 45, "recent")
    assert store.plan_job(scope, 40) is None
    assert not any(n["id"] in outputs for n in store.view(scope)["nodes"])
    store.finish(run)
    job = store.next_job() if store.plan_job(scope, 40) else None
    assert job
    assert job["sources"] == [first, silent] + outputs
    store.job_result(job["id"], "完整一轮")
    store.publish(job)
    expanded = store.expand(scope, first)
    assert [n["id"] for n in expanded["units"][0]["records"]] == [first] + outputs


def test_boundary_round_remains_newer(store):
    scope = store.scope("g", "c")
    old = add(store, scope, 35)
    inp = add(store, scope, 7)
    store.reserve(scope, [inp], "a")
    run = store.begin(scope, "a", store.view(scope, exclude=[inp]), [inp])
    add(store, scope, 8, round_id=run)
    store.finish(run)
    add(store, scope, 32)
    job = store.next_job() if store.plan_job(scope, 40) else None
    assert job["sources"] == [old]


def test_tail_append_and_parallel_results_survive_publish(store):
    scope = store.scope("g", "c")
    old = [add(store, scope, 20) for _ in range(4)]
    job_id = store.plan_job(scope, 40)
    job = store.next_job()
    before = store.view(scope)
    with ThreadPoolExecutor(max_workers=4) as executor:
        fresh = list(
            executor.map(
                lambda i: add(store, scope, 2, str(i), source=f"new:{i}"), range(20)
            )
        )
    store.job_result(job_id, "摘要")
    store.publish(job)
    active = [n["id"] for n in store.view(scope)["nodes"]]
    assert all(i in active for i in fresh + old[2:])
    assert store.publish(job)  # Retry publication is idempotent.
    assert len(before["nodes"]) == 4
    assert len(store.history(scope)["records"]) == 24


def test_invalid_summary_preserves_projection_and_backoff(store):
    scope = store.scope("g", "c")
    [add(store, scope, 20) for _ in range(4)]
    job_id = store.plan_job(scope, 40)
    job = store.next_job()
    original = [n["id"] for n in store.view(scope)["nodes"]]
    for text in ("", "x" * 1000):
        store.job_result(job_id, text)
        with pytest.raises(ValueError):
            store.publish(job)
        assert [n["id"] for n in store.view(scope)["nodes"]] == original
    store.fail_job(job_id, "bad_output")
    assert store.next_job() is None


def test_upper_level_uses_summary_size_not_represented_tokens(store):
    scope = store.scope("g", "c")
    # Each 40-token raw prefix becomes a 10-token summary, so a single
    # summary cannot trigger L1, even though it represents 40 source tokens.
    for batch in range(10):
        while (
            sum(n["tokens"] for n in store.view(scope)["nodes"] if n["level"] == 0) < 80
        ):
            add(store, scope, 20)
        job = store.next_job() if store.plan_job(scope, 40) else None
        store.job_result(job["id"], "摘" * 80)
        store.publish(job)
        upper = store.plan_job(scope, 40)
        if upper:
            next_job = store.next_job()
            if next_job["level"] == 1:
                assert (
                    sum(
                        n["tokens"]
                        for n in store.view(scope)["nodes"]
                        if n["level"] == 1
                    )
                    >= 80
                )
                store.job_result(next_job["id"], "二级摘要")
                store.publish(next_job)
                break
            store.job_result(next_job["id"], "摘" * 80)
            store.publish(next_job)
    assert any(n["level"] == 2 for n in store.view(scope)["nodes"])
    parent = next(n for n in store.view(scope)["nodes"] if n["level"] == 2)
    child = store.describe(scope, parent["id"])["records"][0]
    assert child["level"] == 1
    assert (
        store.expand(scope, child["id"], target="children", limit=10)["records"][0][
            "level"
        ]
        == 0
    )


def test_reservation_conflict_restart_and_session_isolation(store):
    scope = store.scope("g", "c")
    ids = [add(store, scope, 20) for _ in range(4)]
    store.plan_job(scope, 40)
    job = store.next_job()
    store.job_result(job["id"], "摘要")
    store.reserve(scope, [ids[0]], "new_owner")
    assert store.publish(job) is False
    store.release("new_owner")
    jid = store.plan_job(scope, 40)
    store.next_job()
    store.recover()
    retry = store.next_job()
    assert retry["id"] == jid
    other = store.scope("g2", "c")
    with pytest.raises(ValueError):
        store.node(other, ids[0])
    with pytest.raises(ValueError):
        store.expand(other, ids[0])


def test_silent_search_literal_unicode_cursor_and_full_round(store):
    scope = store.scope("g", "c")
    add(store, scope, 1, "你好%_Alice", sender="昵称(User1)", timestamp=100)
    add(store, scope, 1, "你好%_alice", sender="昵称(User1)", timestamp=200)
    assert len(store.search(scope, "%_")["records"]) == 2
    data = store.search(scope, "ALICE", limit=1)
    assert data["next_cursor"]
    assert (
        len(
            store.search(scope, "ALICE", limit=1, cursor=data["next_cursor"])["records"]
        )
        == 1
    )
    with pytest.raises(ValueError):
        store.search(scope, "其他", cursor=data["next_cursor"])
    assert len(store.search(scope, "你好", sender="user1", start=150)["records"]) == 1
    assert not store.search(scope, "不存在")["records"]
    with pytest.raises(ValueError):
        store.search(scope, "")


def test_snapshot_not_reconstructed_and_import_once(store):
    scope = store.scope("g", "c")
    store.import_history(scope, [{"role": "user", "content": "旧记录"}])
    store.import_history(scope, [{"role": "user", "content": "不能重复"}])
    assert len(store.history(scope)["records"]) == 1
    inp = add(store, scope, 1, "new")
    store.reserve(scope, [inp], "r")
    base = store.view(scope, exclude=[inp])
    run = store.begin(scope, "r", base, [inp])
    payload = {
        "contexts": [{"role": "system", "content": "当时人格"}],
        "tools": [{"name": "then"}],
    }
    store.snapshot(scope, run, payload)
    payload["contexts"][0]["content"] = "改了"
    snap = store.request(scope, store.requests(scope, run)[0]["id"])
    assert snap["payload"]["contexts"][0]["content"] == "当时人格"
    store.recover()
    assert store.runs(scope)[0]["status"] == "interrupted"
