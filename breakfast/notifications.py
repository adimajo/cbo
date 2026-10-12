"""E-mail notifications with an iCalendar invitation attached."""

from datetime import timedelta

from django.conf import settings
from django.core.mail import EmailMessage
from django.utils import timezone
from django.utils.formats import date_format
from django.utils.translation import gettext as _
from icalendar import Alarm, Calendar, Event, vCalAddress, vText

from breakfast.models import Breakfast, User

DURATION = timedelta(minutes=30)


def _attendee(user: User, role: str) -> vCalAddress:
    attendee = vCalAddress(f"MAILTO:{user.email}")
    attendee.params["cn"] = vText(f"{user.first_name} - {user.last_name}".strip(" -") or user.username)
    attendee.params["ROLE"] = vText(role)
    return attendee


def _alarm(trigger: timedelta, description: str) -> Alarm:
    alarm = Alarm()
    alarm.add("trigger", trigger)
    alarm.add("action", "display")
    alarm.add("description", description)
    return alarm


def build_invitation(
    breakfast: Breakfast, subject: str, hosts: list[User], participants: list[User], *, reminders: bool = False
) -> bytes:
    """Serialised ``.ics`` event; with ``reminders`` it carries alerts to prepare food (for hosts)."""
    event = Event()
    event.add("uid", f"{breakfast.token}@cbo")
    event.add("dtstamp", timezone.now())
    event.add("summary", subject)
    event.add("dtstart", breakfast.date)
    event.add("dtend", breakfast.date + DURATION)
    event.add("priority", 5)
    organizer = vCalAddress(f"MAILTO:{settings.CBO_ADMIN_EMAIL}")
    organizer.params["cn"] = vText(settings.CBO_ORGANIZER_NAME)
    organizer.params["role"] = vText("CHAIR")
    event["organizer"] = organizer
    if settings.CBO_EVENT_LOCATION:
        event["location"] = vText(settings.CBO_EVENT_LOCATION)
    for host in hosts:
        event.add("attendee", _attendee(host, "REQ-PARTICIPANT"), encode=0)
    for participant in participants:
        event.add("attendee", _attendee(participant, "OPT-PARTICIPANT"), encode=0)
    if reminders:
        event.add_component(_alarm(timedelta(days=-2), _("Two days left to prepare some home-made food")))
        event.add_component(_alarm(timedelta(days=-1), _("The breakfast takes place tomorrow")))
    calendar = Calendar()
    calendar.add("prodid", "-//CBO//breakfast//EN")
    calendar.add("version", "2.0")
    calendar.add_component(event)
    return calendar.to_ical()


def _send(subject: str, body: str, recipients: list[User], invitation: bytes) -> None:
    if not recipients:
        return
    message = EmailMessage(
        subject=subject,
        body=body,
        from_email=settings.CBO_ADMIN_EMAIL,
        to=[user.email for user in recipients],
        cc=[settings.CBO_ADMIN_EMAIL],
        reply_to=[settings.CBO_ADMIN_EMAIL],
    )
    message.attach("breakfast.ics", invitation, "text/calendar")
    message.send()


def _bullets(names: list[str]) -> str:
    return "".join(f"\n\t- {name}" for name in names)


def send_notifications(breakfast: Breakfast) -> None:
    """Tell the hosts what is expected of them and the participants who hosts and the leaderboard are."""
    hosts = breakfast.hosts_users
    participants = breakfast.participants
    when = date_format(timezone.localtime(breakfast.date), "DATETIME_FORMAT")
    subject = _("Breakfast on %(date)s") % {"date": when}

    scores = sorted(breakfast.possible_users, key=lambda user: user.points)
    quarter = len(scores) // 4
    lowest = [user.username for user in scores[:quarter]]
    highest = [user.username for user in scores[len(scores) - quarter :]]

    host_body = _(
        "Hello,\n\n"
        "You have been chosen to host the breakfast taking place on:\n\t%(date)s\n\n"
        "The number of participants expected per host is %(per_host)d.\n"
        "If you can prepare home-made food, you will earn bonus points!\n\n"
        "Kind regards,\n"
        "Chief Breakfast Officer"
    ) % {"date": when, "per_host": len(participants) // max(len(hosts), 1)}

    participant_body = _(
        "Hello,\n\n"
        "You are invited to the breakfast taking place on:\n\t%(date)s\n\n"
        "For your information, the hosts are:%(hosts)s\n\n"
        "The leaderboard now reads:\n"
        "Few points:%(lowest)s\n"
        "Many points:%(highest)s\n\n"
        "CBO"
    ) % {
        "date": when,
        "hosts": _bullets([host.username for host in hosts]),
        "lowest": _bullets(lowest),
        "highest": _bullets(highest),
    }

    _send(subject, host_body, hosts, build_invitation(breakfast, subject, hosts, participants, reminders=True))
    _send(subject, participant_body, participants, build_invitation(breakfast, subject, hosts, participants))
