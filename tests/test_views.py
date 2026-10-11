from django.urls import reverse


def test_pages_require_login(client, db):
    for name in ("home", "breakfast:new", "breakfast:users"):
        response = client.get(reverse(name))
        assert response.status_code == 302 and response.url.startswith("/login/")


def test_login_and_logout(client, team):
    assert client.post(reverse("login"), {"username": "alice", "password": "pw"}).status_code == 302
    assert client.get(reverse("home")).status_code == 200
    assert client.post(reverse("logout")).status_code == 302
    assert client.get(reverse("home")).status_code == 302


def test_logout_needs_post(logged_client):
    assert logged_client.get(reverse("logout")).status_code == 405


def test_wrong_password_shows_error(client, team):
    response = client.post(reverse("login"), {"username": "alice", "password": "nope"})
    assert response.status_code == 200
    assert "didn't match" in response.content.decode()


def test_home_and_new_pages_render(logged_client):
    assert "Hello alice!" in logged_client.get(reverse("home")).content.decode()
    page = logged_client.get(reverse("breakfast:new")).content.decode()
    assert 'id="breakfast-app"' in page and "breakfast/new.js" in page


def test_calendar_navigation_ignores_garbage(logged_client):
    assert logged_client.get(reverse("home"), {"year": "x", "month": "99"}).status_code == 200
    page = logged_client.get(reverse("home"), {"year": 2024, "month": 12}).content.decode()
    assert "December 2024" in page and "year=2025&amp;month=1" in page


def test_user_list_filters_by_team_and_sorts_by_points(logged_client, make_user):
    make_user("zed", team="Ops", points=7)
    make_user("yan", team="Ops", points=9)
    page = logged_client.get(reverse("breakfast:users"), {"team": "Ops"}).content.decode()
    assert page.index("yan") < page.index("zed")
    assert "alice" not in page.split("<tbody>")[1]


def test_user_detail_lists_confirmed_breakfasts_only(logged_client, team, make_breakfast):
    make_breakfast(hosts=[team[1]])
    make_breakfast(hosts=[team[1]], confirmed=False)
    response = logged_client.get(reverse("breakfast:user-detail", args=[team[1].pk]))
    assert len(response.context["rows"]) == 1
    assert response.context["rows"][0]["points"] == 3
    assert "Signed in as alice" in response.content.decode()  # the viewer, not the profile


def test_language_switch_translates_the_interface(logged_client):
    assert "Users" in logged_client.get(reverse("home")).content.decode()
    response = logged_client.post(reverse("set_language"), {"language": "fr", "next": reverse("home")})
    assert response.status_code == 302
    page = logged_client.get(reverse("home")).content.decode()
    assert "Utilisateurs" in page and "Bonjour alice !" in page and '<html lang="fr">' in page
    logged_client.post(reverse("set_language"), {"language": "en", "next": "/"})
    assert "Hello alice!" in logged_client.get(reverse("home")).content.decode()


def test_new_page_strings_are_translated_for_the_script(logged_client):
    logged_client.post(reverse("set_language"), {"language": "fr", "next": "/"})
    page = logged_client.get(reverse("breakfast:new")).content.decode()
    assert "Confirmer et notifier" in page and "Enregistrement…" in page


def test_api_errors_follow_the_language(logged_client, team):
    from breakfast.models import Breakfast  # noqa: F401

    logged_client.post(reverse("set_language"), {"language": "fr", "next": "/"})
    import json

    draft = logged_client.post(reverse("breakfast:draft-create"), "{}", content_type="application/json").json()
    response = logged_client.post(
        reverse("breakfast:draft-update", kwargs={"token": draft["token"]}),
        json.dumps({**draft, "action": "confirm"}),
        content_type="application/json",
    )
    assert "au moins un organisateur" in response.json()["error"]


def test_admin_is_usable(client, make_user):
    admin = make_user("root", is_staff=True, is_superuser=True)
    client.force_login(admin)
    for name in ("user", "breakfast", "host", "userpointadjustment", "globalpointadjustment"):
        assert client.get(reverse(f"admin:breakfast_{name}_changelist")).status_code == 200
    assert client.get(reverse("admin:breakfast_user_add")).status_code == 200
