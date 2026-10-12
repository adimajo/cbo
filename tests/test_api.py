import json
from datetime import date

from django.core import mail
from django.urls import reverse

from breakfast.models import Breakfast


def post(client, name, data=None, **kwargs):
    return client.post(reverse(name, **kwargs), json.dumps(data or {}), content_type="application/json")


def update(client, token, data):
    return post(client, "breakfast:draft-update", data, kwargs={"token": token})


def state_for(draft, **changes):
    return {"date": draft["date"], "include_optional_team": True, "absentees": [], "hosts": [], **changes}


def test_api_requires_authentication(client, db):
    response = post(client, "breakfast:draft-create")
    assert response.status_code == 401
    assert Breakfast.objects.count() == 0


def test_create_draft_defaults_to_tomorrow_morning(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    assert draft["date"].endswith("T09:00")
    assert draft["confirmed"] is False
    assert {user["email"] for user in draft["possible_users"]} == {f"{u.username}@example.com" for u in team}
    assert draft["participant_count"] == 4


def test_update_applies_date_absentees_and_hosts(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    response = update(
        logged_client,
        draft["token"],
        state_for(
            draft,
            date="2024-05-14T08:30",
            absentees=["dave@example.com"],
            hosts=[{"email": "alice@example.com", "homemade": True}],
        ),
    ).json()
    assert response["date"] == "2024-05-14T08:30"
    assert response["absentees"] == ["dave@example.com"]
    assert [(h["email"], h["homemade"]) for h in response["hosts"]] == [("alice@example.com", True)]
    assert response["participant_count"] == 2


def test_absent_people_cannot_host(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    response = update(
        logged_client,
        draft["token"],
        state_for(draft, absentees=["bob@example.com"], hosts=[{"email": "bob@example.com"}]),
    ).json()
    assert response["hosts"] == []


def test_unknown_people_are_ignored(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    response = update(logged_client, draft["token"], state_for(draft, hosts=[{"email": "nobody@example.com"}])).json()
    assert response["hosts"] == []


def test_draw_adds_a_host(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    response = update(logged_client, draft["token"], {**state_for(draft), "action": "draw"}).json()
    assert len(response["hosts"]) == 1
    assert response["participant_count"] == 3


def test_draw_with_nobody_left_is_an_error(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    everyone = [f"{user.username}@example.com" for user in team]
    response = update(logged_client, draft["token"], {**state_for(draft, absentees=everyone), "action": "draw"})
    assert response.status_code == 400


def test_add_designated_host(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    response = update(
        logged_client, draft["token"], {**state_for(draft), "action": "add_host", "email": "carol@example.com"}
    ).json()
    assert [h["email"] for h in response["hosts"]] == ["carol@example.com"]


def test_cannot_add_someone_already_hosting(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    body = state_for(draft, hosts=[{"email": "carol@example.com"}])
    response = update(logged_client, draft["token"], {**body, "action": "add_host", "email": "carol@example.com"})
    assert response.status_code == 400


def test_invalid_date_and_action(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    assert update(logged_client, draft["token"], state_for(draft, date="yesterday")).status_code == 400
    assert update(logged_client, draft["token"], {**state_for(draft), "action": "explode"}).status_code == 400
    bad = logged_client.post(
        reverse("breakfast:draft-update", kwargs={"token": draft["token"]}), "{", content_type="application/json"
    )
    assert bad.status_code == 400


def test_unknown_token_is_a_404(logged_client, team):
    assert update(logged_client, "0" * 16, {}).status_code == 404


def test_confirm_needs_a_host(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    response = update(logged_client, draft["token"], {**state_for(draft), "action": "confirm"})
    assert response.status_code == 400
    assert not Breakfast.objects.get(token=draft["token"]).is_confirmed


def test_confirm_sends_two_mails_and_locks_the_draft(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    body = state_for(draft, date="2024-05-14T09:00", hosts=[{"email": "alice@example.com", "homemade": True}])
    response = update(logged_client, draft["token"], {**body, "action": "confirm"}).json()

    assert response["confirmed"] is True
    host_mail, participant_mail = mail.outbox
    assert host_mail.to == ["alice@example.com"]
    assert sorted(participant_mail.to) == ["bob@example.com", "carol@example.com", "dave@example.com"]
    assert host_mail.subject == participant_mail.subject
    assert "BEGIN:VALARM" in host_mail.attachments[0].content
    assert "BEGIN:VALARM" not in participant_mail.attachments[0].content
    assert "alice" in participant_mail.body

    team[0].refresh_from_db()
    assert team[0].points == 4.5  # three participants pay one point each, +50% for home-made

    locked = update(logged_client, draft["token"], state_for(draft))
    assert locked.status_code == 409


def test_failing_smtp_keeps_the_draft(logged_client, team, settings, monkeypatch):
    def broken(self, *args, **kwargs):
        raise ConnectionRefusedError("smtp down")

    monkeypatch.setattr("django.core.mail.message.EmailMessage.send", broken)
    draft = post(logged_client, "breakfast:draft-create").json()
    body = state_for(draft, hosts=[{"email": "alice@example.com"}])
    response = update(logged_client, draft["token"], {**body, "action": "confirm"})
    assert response.status_code == 502
    assert "smtp down" in response.json()["error"]
    assert not Breakfast.objects.get(token=draft["token"]).is_confirmed


def test_confirmed_breakfast_shows_on_calendar(logged_client, team):
    draft = post(logged_client, "breakfast:draft-create").json()
    body = state_for(draft, date="2024-05-14T09:00", hosts=[{"email": "alice@example.com"}])
    update(logged_client, draft["token"], {**body, "action": "confirm"})
    page = logged_client.get(reverse("home"), {"year": 2024, "month": 5}).content.decode()
    assert "May 2024" in page and "<li>alice</li>" in page
    assert date(2024, 5, 14).day == 14
