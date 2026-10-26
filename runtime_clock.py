"""Current local time and the mainland China work/rest calendar."""

from datetime import datetime

from chinese_calendar import Holiday, get_holiday_detail


def runtime_time(current: datetime) -> str:
    """Use published calendar data, including make-up workdays."""
    weekday = "周" + "一二三四五六日"[current.weekday()]
    try:
        resting, holiday = get_holiday_detail(current.date())
    except NotImplementedError:
        kind = "节假日安排未知"
    else:
        if resting:
            kind = (
                f"节假日（{Holiday(holiday).chinese}）"
                if holiday
                else "休息日（周末）"
            )
        else:
            kind = "法定工作日（调休补班）" if holiday else "法定工作日"
    return f"时间：{current:%Y-%m-%d %H:%M:%S}；{weekday}；{kind}"
