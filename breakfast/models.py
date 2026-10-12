"""Data model: users with a points balance, breakfasts, their hosts and manual point adjustments."""

import secrets
from datetime import date

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.db.models import Q
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

#: Extra points a host earns per participant when the breakfast is home-made.
HOMEMADE_BONUS = 0.5


def _generate_token() -> str:
    return secrets.token_hex(8)


def _today() -> date:
    return timezone.localdate()


class User(AbstractUser):
    arrival_date = models.DateField(_("arrival date"), default=_today)
    departure_date = models.DateField(_("departure date"), blank=True, null=True)
    bio = models.TextField(_("biography"), max_length=500, blank=True)
    specialty = models.CharField(_("specialty"), max_length=30, blank=True)
    team = models.CharField(_("team"), max_length=40, blank=True)
    points = models.FloatField(
        _("points"),
        default=0,
        help_text=_("Balance, recomputed from the confirmed breakfasts. Low balances are drawn first."),
    )

    @property
    def last_hosted_date(self):
        """Date of the most recent breakfast this user hosted, or ``None``."""
        host = self.hostings.filter(breakfast__confirmed_at__isnull=False).order_by("-breakfast__date").first()
        return host.breakfast.date if host else None

    def to_dict(self) -> dict:
        return {
            "email": self.email,
            "username": self.username,
            "name": self.get_full_name() or self.username,
        }


class BreakfastQuerySet(models.QuerySet):
    def confirmed(self):
        return self.filter(confirmed_at__isnull=False)

    def drafts(self):
        return self.filter(confirmed_at__isnull=True)


class Breakfast(models.Model):
    created_at = models.DateTimeField(_("creation date"), auto_now_add=True)
    updated_at = models.DateTimeField(_("last update"), auto_now=True)
    date = models.DateTimeField(_("breakfast date"))
    token = models.CharField(_("token"), max_length=32, unique=True, default=_generate_token, editable=False)
    confirmed_at = models.DateTimeField(
        _("confirmed at"),
        null=True,
        blank=True,
        help_text=_("Set once the notifications were sent. Only confirmed breakfasts count towards points."),
    )
    include_optional_team = models.BooleanField(_("optional team takes part"), default=True)
    absentees = models.ManyToManyField(User, verbose_name=_("absentees"), related_name="absences", blank=True)

    objects = BreakfastQuerySet.as_manager()

    class Meta:
        ordering = ["-date"]
        verbose_name = _("breakfast")
        verbose_name_plural = _("breakfasts")

    def __str__(self) -> str:
        return timezone.localtime(self.date).strftime("%Y-%m-%d %H:%M")

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed_at is not None

    @property
    def possible_users(self):
        """Everyone who could attend: arrived, not departed, active, and not the service account."""
        day = timezone.localtime(self.date).date()
        queryset = User.objects.filter(is_active=True, arrival_date__lte=day).filter(
            Q(departure_date__isnull=True) | Q(departure_date__gt=day)
        )
        if settings.CBO_OPTIONAL_TEAM and not self.include_optional_team:
            queryset = queryset.exclude(team=settings.CBO_OPTIONAL_TEAM)
        return queryset.exclude(email=settings.CBO_ADMIN_EMAIL).order_by("username")

    @property
    def hosts_users(self) -> list[User]:
        return [host.user for host in self.hosts.select_related("user")]

    @property
    def participants(self) -> list[User]:
        excluded = {user.pk for user in self.hosts_users} | set(self.absentees.values_list("pk", flat=True))
        return [user for user in self.possible_users if user.pk not in excluded]

    @property
    def absentee_count(self) -> int:
        return self.absentees.count()

    @property
    def participant_count(self) -> int:
        return len(self.participants)

    @property
    def host_count(self) -> int:
        return self.hosts.count()

    def points(self) -> dict[int, float]:
        """Points each user earned or lost at this breakfast, keyed by user id.

        Participants pay one point each; hosts share what was paid, with a bonus for home-made food.
        """
        hosts = list(self.hosts.select_related("user"))
        if not hosts:
            return {}
        participants = self.participants
        share = len(participants) / len(hosts)
        result = {user.pk: -1.0 for user in participants}
        for host in hosts:
            result[host.user_id] = share * (1 + HOMEMADE_BONUS * host.homemade)
        return result


class Host(models.Model):
    created_at = models.DateTimeField(_("creation date"), auto_now_add=True)
    updated_at = models.DateTimeField(_("last update"), auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="hostings", verbose_name=_("host"))
    breakfast = models.ForeignKey(
        Breakfast, on_delete=models.CASCADE, related_name="hosts", verbose_name=_("breakfast")
    )
    homemade = models.BooleanField(_("home-made"), default=False)

    class Meta:
        verbose_name = _("host")
        verbose_name_plural = _("hosts")
        constraints = [models.UniqueConstraint(fields=["user", "breakfast"], name="unique_host_per_breakfast")]

    def __str__(self) -> str:
        return f"{self.user} - {self.breakfast}"

    def to_dict(self) -> dict:
        return {**self.user.to_dict(), "homemade": self.homemade}


class UserPointAdjustment(models.Model):
    """Manual correction applied to a single user."""

    created_at = models.DateTimeField(_("creation date"), auto_now_add=True)
    updated_at = models.DateTimeField(_("last update"), auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="adjustments", verbose_name=_("user"))
    points = models.FloatField(_("points"), default=0)

    class Meta:
        verbose_name = _("point adjustment (user)")
        verbose_name_plural = _("point adjustments (user)")

    def __str__(self) -> str:
        return f"{self.user}: {self.points:+g}"


class GlobalPointAdjustment(models.Model):
    """Manual correction applied to every user."""

    created_at = models.DateTimeField(_("creation date"), auto_now_add=True)
    updated_at = models.DateTimeField(_("last update"), auto_now=True)
    points = models.FloatField(_("points"), default=0)

    class Meta:
        verbose_name = _("point adjustment (everyone)")
        verbose_name_plural = _("point adjustments (everyone)")

    def __str__(self) -> str:
        return f"{self.points:+g}"
