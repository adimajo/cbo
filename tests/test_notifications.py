from django.core import mail
from django.utils import translation

from breakfast.notifications import build_invitation, send_notifications


def test_mails_are_sent_in_the_active_language(team, make_breakfast):
    breakfast = make_breakfast(hosts=[team[0]])
    with translation.override("fr"):
        send_notifications(breakfast)
    host_mail, participant_mail = mail.outbox
    assert host_mail.subject.startswith("Petit-déjeuner du")
    assert "Vous avez été désigné" in host_mail.body
    assert "Peu de points" in participant_mail.body

    mail.outbox.clear()
    with translation.override("en"):
        send_notifications(breakfast)
    assert mail.outbox[0].subject.startswith("Breakfast on")
    assert "You have been chosen to host" in mail.outbox[0].body


def test_mails_are_copied_to_the_shared_mailbox(team, make_breakfast, settings):
    send_notifications(make_breakfast(hosts=[team[0]]))
    for message in mail.outbox:
        assert message.from_email == settings.CBO_ADMIN_EMAIL
        assert message.cc == message.reply_to == [settings.CBO_ADMIN_EMAIL]


def test_no_mail_for_an_empty_audience(team, make_breakfast):
    send_notifications(make_breakfast(hosts=[team[0]], absentees=team[1:]))
    assert len(mail.outbox) == 1  # only the host is told


def test_invitation_content(team, make_breakfast, settings):
    settings.CBO_EVENT_LOCATION = "Kitchen"
    breakfast = make_breakfast(hosts=[team[0]])
    ics = build_invitation(breakfast, "Subject", [team[0]], team[1:]).decode()
    assert f"UID:{breakfast.token}@cbo" in ics and "DTSTAMP" in ics
    assert "LOCATION:Kitchen" in ics
    assert ics.count("REQ-PARTICIPANT") == 1 and ics.count("OPT-PARTICIPANT") == 3
    assert "VALARM" not in ics
