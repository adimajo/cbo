"""Month grid showing who hosts which day."""

import calendar
from dataclasses import dataclass, field
from datetime import date

from django.utils import timezone
from django.utils.dates import MONTHS, WEEKDAYS_ABBR

from breakfast.models import Breakfast


@dataclass
class Day:
    date: date
    in_month: bool
    breakfasts: list[list[str]] = field(default_factory=list)  # one list of host names per breakfast


@dataclass
class MonthGrid:
    year: int
    month: int
    title: str
    weekdays: list[str]
    weeks: list[list[Day]]
    previous: tuple[int, int]
    next: tuple[int, int]


def _shift(year: int, month: int, offset: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + offset
    return index // 12, index % 12 + 1


def month_grid(year: int, month: int) -> MonthGrid:
    """Build the grid for ``year``/``month`` (Monday first) with the confirmed breakfasts filled in."""
    weeks = [
        [Day(day, day.month == month) for day in week]
        for week in calendar.Calendar(firstweekday=0).monthdatescalendar(year, month)
    ]
    by_day = {day.date: day for week in weeks for day in week}
    first, last = weeks[0][0].date, weeks[-1][-1].date
    breakfasts = (
        Breakfast.objects.confirmed()
        .filter(date__date__gte=first, date__date__lte=last)
        .prefetch_related("hosts__user")
        .order_by("date")
    )
    for breakfast in breakfasts:
        day = by_day.get(timezone.localtime(breakfast.date).date())
        if day:
            day.breakfasts.append([host.user.username for host in breakfast.hosts.all()])
    return MonthGrid(
        year=year,
        month=month,
        title=f"{MONTHS[month]} {year}",
        weekdays=[str(WEEKDAYS_ABBR[index]) for index in range(7)],
        weeks=weeks,
        previous=_shift(year, month, -1),
        next=_shift(year, month, 1),
    )
