"""Pages and the JSON API used by the "new breakfast" page."""

import json
from datetime import datetime, time, timedelta

from django.contrib.auth.mixins import LoginRequiredMixin
from django.db import transaction
from django.http import HttpRequest, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.translation import gettext as _
from django.views import View
from django.views.generic import DetailView, ListView, TemplateView

from breakfast import services
from breakfast.calendar import month_grid
from breakfast.models import Breakfast, Host, User
from breakfast.notifications import send_notifications


class CalendarMixin:
    """Adds a month grid to the context, driven by the ``year`` and ``month`` query parameters."""

    def get_calendar(self):
        today = timezone.localdate()
        try:
            year, month = int(self.request.GET["year"]), int(self.request.GET["month"])
            if not (1 <= month <= 12 and 1900 <= year <= 2200):
                raise ValueError
        except (KeyError, ValueError):
            year, month = today.year, today.month
        return month_grid(year, month)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["calendar"] = self.get_calendar()
        context["today"] = timezone.localdate()
        return context


class HomeView(LoginRequiredMixin, CalendarMixin, TemplateView):
    template_name = "breakfast/home.html"


class UserListView(LoginRequiredMixin, ListView):
    template_name = "breakfast/user_list.html"
    context_object_name = "users"
    paginate_by = 50

    def get_queryset(self):
        queryset = User.objects.order_by("-points", "username")
        team = self.request.GET.get("team")
        return queryset.filter(team=team) if team else queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["teams"] = User.objects.exclude(team="").order_by("team").values_list("team", flat=True).distinct()
        context["selected_team"] = self.request.GET.get("team", "")
        return context


class UserDetailView(LoginRequiredMixin, DetailView):
    template_name = "breakfast/user_detail.html"
    model = User
    context_object_name = "profile"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        rows = []
        for breakfast in Breakfast.objects.confirmed().prefetch_related("hosts__user", "absentees"):
            rows.append(
                {
                    "breakfast": breakfast,
                    "absentees": breakfast.absentee_count,
                    "participants": breakfast.participant_count,
                    "hosts": breakfast.host_count,
                    "points": round(breakfast.points().get(self.object.pk, 0), 2),
                }
            )
        context["rows"] = rows
        return context


class NewBreakfastView(LoginRequiredMixin, CalendarMixin, TemplateView):
    template_name = "breakfast/new.html"


# ---------------------------------------------------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def draft_state(breakfast: Breakfast) -> dict:
    """Everything the frontend needs to render a breakfast being prepared."""
    return {
        "token": breakfast.token,
        "date": timezone.localtime(breakfast.date).strftime("%Y-%m-%dT%H:%M"),
        "include_optional_team": breakfast.include_optional_team,
        "confirmed": breakfast.is_confirmed,
        "possible_users": [user.to_dict() for user in breakfast.possible_users],
        "absentees": [user.email for user in breakfast.absentees.all()],
        "hosts": [host.to_dict() for host in breakfast.hosts.select_related("user")],
        "participant_count": breakfast.participant_count,
    }


class ApiView(View):
    """Base class: JSON in, JSON out, authentication errors reported as JSON rather than redirects."""

    def dispatch(self, request: HttpRequest, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": _("Authentication required.")}, status=401)
        try:
            return super().dispatch(request, *args, **kwargs)
        except ApiError as error:
            return JsonResponse({"error": str(error)}, status=error.status)

    @staticmethod
    def payload(request: HttpRequest) -> dict:
        try:
            data = json.loads(request.body or b"{}")
        except json.JSONDecodeError as error:
            raise ApiError(_("Invalid JSON.")) from error
        if not isinstance(data, dict):
            raise ApiError(_("Invalid JSON."))
        return data


class DraftCreateView(ApiView):
    http_method_names = ["post"]

    def post(self, request: HttpRequest):
        tomorrow = timezone.localdate() + timedelta(days=1)
        date = timezone.make_aware(datetime.combine(tomorrow, time(9, 0)))
        return JsonResponse(draft_state(services.create_draft(date)))


class DraftUpdateView(ApiView):
    """Apply the state sent by the page, then an optional action (``draw``, ``add_host`` or ``confirm``)."""

    http_method_names = ["post"]

    def post(self, request: HttpRequest, token: str):
        breakfast = get_object_or_404(Breakfast, token=token)
        if breakfast.is_confirmed:
            raise ApiError(_("This breakfast is already confirmed."), status=409)
        data = self.payload(request)
        action = data.get("action", "update")
        if action not in {"update", "draw", "add_host", "confirm"}:
            raise ApiError(_("Unknown action."))

        with transaction.atomic():
            self.apply_state(breakfast, data)
            if action == "draw" and services.draw_host(breakfast) is None:
                raise ApiError(_("Nobody is left to host."))
            if action == "add_host":
                self.add_host(breakfast, data.get("email"))
            if action == "confirm" and not breakfast.hosts.exists():
                raise ApiError(_("Choose at least one host before confirming."))

        if action == "confirm":
            self.confirm(breakfast)
        return JsonResponse(draft_state(breakfast))

    @staticmethod
    def apply_state(breakfast: Breakfast, data: dict) -> None:
        try:
            date = datetime.fromisoformat(str(data["date"]))
        except (KeyError, ValueError) as error:
            raise ApiError(_("Invalid date.")) from error
        if timezone.is_naive(date):
            date = timezone.make_aware(date)
        breakfast.date = date
        breakfast.include_optional_team = bool(data.get("include_optional_team", True))
        breakfast.save()

        possible = {user.email: user for user in breakfast.possible_users}
        absentees = {possible[email] for email in data.get("absentees", []) if email in possible}
        breakfast.absentees.set(absentees)

        wanted = {
            item["email"]: bool(item.get("homemade"))
            for item in data.get("hosts", [])
            if isinstance(item, dict) and item.get("email") in possible
        }
        wanted = {email: homemade for email, homemade in wanted.items() if possible[email] not in absentees}
        breakfast.hosts.exclude(user__email__in=list(wanted)).delete()
        for email, homemade in wanted.items():
            Host.objects.update_or_create(breakfast=breakfast, user=possible[email], defaults={"homemade": homemade})

    @staticmethod
    def add_host(breakfast: Breakfast, email) -> None:
        user = next((user for user in breakfast.participants if user.email == email), None)
        if user is None:
            raise ApiError(_("This person cannot host this breakfast."))
        Host.objects.create(breakfast=breakfast, user=user)

    @staticmethod
    def confirm(breakfast: Breakfast) -> None:
        services.refresh_points()
        try:
            send_notifications(breakfast)
        except OSError as error:  # smtplib.SMTPException is an OSError subclass
            raise ApiError(
                _("The notifications could not be sent: %(error)s") % {"error": error}, status=502
            ) from error
        services.confirm(breakfast)
