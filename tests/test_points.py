import pytest

from breakfast import services
from breakfast.models import GlobalPointAdjustment, UserPointAdjustment


def test_participants_pay_one_point_and_host_collects(team, make_breakfast):
    alice, bob, carol, dave = team
    breakfast = make_breakfast(hosts=[alice])
    assert breakfast.points() == {bob.pk: -1, carol.pk: -1, dave.pk: -1, alice.pk: 3}


def test_homemade_bonus_and_shared_hosting(team, make_breakfast):
    alice, bob, carol, dave = team
    breakfast = make_breakfast(hosts=[alice, bob], homemade=[bob])
    # 2 participants, 2 hosts: each host collects 1 point, +50% when home-made
    assert breakfast.points() == {carol.pk: -1, dave.pk: -1, alice.pk: 1, bob.pk: 1.5}


def test_no_host_means_no_points(team, make_breakfast):
    assert make_breakfast().points() == {}


def test_absentees_neither_pay_nor_count(team, make_breakfast):
    alice, bob, carol, dave = team
    breakfast = make_breakfast(hosts=[alice], absentees=[dave])
    assert breakfast.points() == {bob.pk: -1, carol.pk: -1, alice.pk: 2}


def test_users_who_arrived_later_or_left_are_ignored(team, make_user, make_breakfast):
    from datetime import date

    make_user("late", arrival_date=date(2030, 1, 1))
    make_user("gone", departure_date=date(2024, 1, 1))
    make_user("inactive", is_active=False)
    breakfast = make_breakfast(hosts=[team[0]])
    assert breakfast.participant_count == 3


def test_service_account_and_optional_team(team, make_user, make_breakfast, settings):
    settings.CBO_OPTIONAL_TEAM = "Validation"
    make_user("service", email=settings.CBO_ADMIN_EMAIL)
    make_user("val", team="Validation")
    with_team = make_breakfast(hosts=[team[0]], include_optional_team=True)
    without_team = make_breakfast(hosts=[team[0]], include_optional_team=False)
    assert with_team.participant_count == 4
    assert without_team.participant_count == 3


def test_refresh_points_counts_confirmed_breakfasts_and_adjustments(team, make_breakfast):
    alice, bob, carol, dave = team
    make_breakfast(hosts=[alice])
    make_breakfast(hosts=[bob], confirmed=False)  # drafts must not count
    GlobalPointAdjustment.objects.create(points=10)
    UserPointAdjustment.objects.create(user=carol, points=-4)

    services.refresh_points()

    points = {user.username: user.points for user in type(alice).objects.all()}
    assert points == {"alice": 13, "bob": 9, "carol": 5, "dave": 9}


@pytest.mark.django_db
def test_fractional_points_are_kept(team, make_breakfast):
    make_breakfast(hosts=[team[0], team[1]], homemade=[team[1]])
    services.refresh_points()
    team[1].refresh_from_db()
    assert team[1].points == 1.5
