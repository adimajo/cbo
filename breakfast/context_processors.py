"""Template context shared by every page."""

from django.conf import settings


def site(request):
    return {
        "admin_email": settings.CBO_ADMIN_EMAIL,
        "optional_team": settings.CBO_OPTIONAL_TEAM,
    }
