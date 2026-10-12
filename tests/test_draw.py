import random
from collections import Counter

from breakfast import services
from breakfast.models import GlobalPointAdjustment, UserPointAdjustment


def test_only_lowest_balance_is_drawn_when_everybody_is_positive(team, make_breakfast):
    alice, bob, carol, dave = team
    GlobalPointAdjustment.objects.create(points=5)
    UserPointAdjustment.objects.create(user=carol, points=-3)  # lowest, still positive
    breakfast = make_breakfast(confirmed=False)
    for seed in range(20):
        host = services.draw_host(breakfast, random.Random(seed))
        assert host.user == carol
        host.delete()


def test_candidates_are_weighted_by_how_negative_they_are(team, make_breakfast):
    alice, bob, carol, dave = team
    UserPointAdjustment.objects.create(user=alice, points=-1)
    UserPointAdjustment.objects.create(user=bob, points=-3)
    breakfast = make_breakfast(confirmed=False)
    rng = random.Random(0)
    counts = Counter()
    for _ in range(600):
        host = services.draw_host(breakfast, rng)
        counts[host.user.username] += 1
        host.delete()
    # alice and bob (-1, -3) weigh 2 and 4 against 1 each for carol and dave (0): shares 2/8, 4/8, 1/8, 1/8
    assert counts["bob"] > counts["alice"] > counts["carol"]
    assert counts["bob"] > 250 and counts["carol"] > 40 and counts["dave"] > 40


def test_hosts_and_absentees_are_never_drawn(team, make_breakfast):
    alice, bob, carol, dave = team
    breakfast = make_breakfast(confirmed=False, hosts=[alice], absentees=[bob, carol])
    assert services.draw_host(breakfast).user == dave


def test_draw_returns_none_when_nobody_is_left(team, make_breakfast):
    breakfast = make_breakfast(confirmed=False, absentees=team)
    assert services.draw_host(breakfast) is None


def test_stale_drafts_are_discarded(team, make_breakfast, settings):
    from datetime import timedelta

    from django.utils import timezone

    from breakfast.models import Breakfast

    stale = make_breakfast(confirmed=False)
    Breakfast.objects.filter(pk=stale.pk).update(
        created_at=timezone.now() - timedelta(hours=settings.CBO_DRAFT_TTL_HOURS + 1)
    )
    confirmed = make_breakfast()
    services.create_draft(timezone.now())
    assert not Breakfast.objects.filter(pk=stale.pk).exists()
    assert Breakfast.objects.filter(pk=confirmed.pk).exists()
