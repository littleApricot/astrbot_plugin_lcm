"""Check current-time hints against published holiday and make-up dates."""

from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from chinese_calendar import holidays

from astrbot_plugin_lcm import context_policy
from astrbot_plugin_lcm.runtime_clock import runtime_time
from astrbot_plugin_lcm.tests.test_integration import owner  # noqa: F401


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-10-26 18:04:12", "周一；法定工作日"),
        ("2026-10-25 18:04:12", "周日；休息日（周末）"),
        ("2026-10-01 18:04:12", "周四；节假日（国庆节）"),
        ("2026-10-10 18:04:12", "周六；法定工作日（调休补班）"),
        ("2026-09-20 18:04:12", "周日；法定工作日（调休补班）"),
    ],
)
def test_work_rest_calendar(stamp, expected):
    assert runtime_time(datetime.fromisoformat(stamp)) == f"时间：{stamp}；{expected}"


def test_unknown_year_is_not_guessed_from_weekday():
    current = datetime(max(holidays).year + 1, 1, 1)
    assert runtime_time(current).endswith("；节假日安排未知")


def test_temporary_hint_uses_configured_date_and_stays_out_of_system(
    owner, monkeypatch
):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 30, 16, 4, 12, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(context_policy, "datetime", Clock)
    owner.context = SimpleNamespace(
        get_config=lambda **kw: {"timezone": "Asia/Shanghai"}
    )
    extras = {}
    event = SimpleNamespace(
        unified_msg_origin="group",
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
    )
    rules = owner.context_rules()
    expected = runtime_time(Clock.now(ZoneInfo("Asia/Shanghai")))
    assert expected == "时间：2026-10-01 00:04:12；周四；节假日（国庆节）"
    assert rules.temporary(event) == (
        "[本轮临时提示｜来源：astrbot.runtime]\n"
        + expected
        + "\n[/本轮临时提示]"
    )
    assert "时间：" not in rules.system(event, "人设")
