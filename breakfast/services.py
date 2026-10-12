"""Business logic: points bookkeeping, the weighted draw of hosts and the draft life cycle."""

import random
import secrets
from collections import defaultdict
from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.db.models import Sum
from django.utils import timezone

from breakfast.models import Breakfast, GlobalPointAdjustment, Host, User, UserPointAdjustment


def compute_points() -> dict[int, float]:
    """Points of every user: manual adjustments plus the outcome of every confirmed breakfast."""
    global_delta = GlobalPointAdjustment.objects.aggregate(total=Sum("points"))["total"] or 0
    totals: dict[int, float] = defaultdict(float)
    totals.update({pk: global_delta for pk in User.objects.values_list("pk", flat=True)})
    for adjustment in UserPointAdjustment.objects.all():
        totals[adjustment.user_id] += adjustment.points
    for breakfast in Breakfast.objects.confirmed():
        for user_id, delta in breakfast.points().items():
            totals[user_id] += delta
    return totals


def refresh_points() -> None:
    """Store the freshly computed balance on each user."""
    totals = compute_points()
    users = list(User.objects.all())
    changed = []
    for user in users:
        if user.points != totals[user.pk]:
            user.points = totals[user.pk]
            changed.append(user)
    User.objects.bulk_update(changed, ["points"])


def draw_host(breakfast: Breakfast, rng: random.Random | None = None) -> Host | None:
    """Add a randomly drawn host to ``breakfast`` and return it (``None`` if nobody is left).

    Only the users with the lowest balance are candidates: those at or below zero when anyone is, the single
    lowest otherwise. Among them, the lower the balance the likelier the draw.
    """
    rng = rng or secrets.SystemRandom()
    refresh_points()
    balances = {user.pk: user.points for user in breakfast.participants}
    if not balances:
        return None
    floor = max(min(balances.values()), 0)
    candidates = {pk: points for pk, points in balances.items() if points <= floor}
    ids = list(candidates)
    weights = [floor + 1 - candidates[pk] for pk in ids]
    chosen = rng.choices(ids, weights=weights, k=1)[0]
    return Host.objects.create(breakfast=breakfast, user_id=chosen)


def create_draft(date) -> Breakfast:
    """Create a draft breakfast, discarding stale drafts on the way."""
    cutoff = timezone.now() - timedelta(hours=settings.CBO_DRAFT_TTL_HOURS)
    Breakfast.objects.drafts().filter(created_at__lt=cutoff).delete()
    return Breakfast.objects.create(date=date)


@transaction.atomic
def confirm(breakfast: Breakfast) -> None:
    """Mark ``breakfast`` as confirmed and update everybody's balance."""
    breakfast.confirmed_at = timezone.now()
    breakfast.save(update_fields=["confirmed_at", "updated_at"])
    refresh_points()
