How it works
============

Points
------

The balance of a user is the sum of

* the *global adjustments* (applied to everybody),
* the *user adjustments* (applied to one person), and
* the outcome of every **confirmed** breakfast:

  * each participant pays one point;
  * the points paid are shared between the hosts, who each earn
    ``participants / hosts``, increased by 50% when the food is home-made;
  * absentees and people who had not arrived yet (or already left) neither pay nor earn.

Drawing a host
--------------

Among the people who could host (participants who are not already hosts), only those with the lowest balance are
candidates: the ones at or below zero if any, the single lowest otherwise. The draw is weighted so that the lower the
balance, the likelier the person is picked.

Life cycle of a breakfast
-------------------------

Opening the *New breakfast* page creates a **draft**. Every change (date, absentees, hosts) is saved on the server as
it happens. *Confirm and notify* e-mails the hosts and the participants, with a calendar invitation, and locks the
breakfast: from then on it counts towards points. Drafts that are never confirmed are discarded after
``CBO_DRAFT_TTL_HOURS``.

Translations
------------

User-facing strings go through Django's translation machinery; catalogues live in ``breakfast/locale``. To add or
update a language, use ``python manage.py makemessages -l <code>`` and ``python manage.py compilemessages`` (needs GNU
gettext), or edit the ``.po`` file and run ``python scripts/compile_messages.py``. Then add the language to
``LANGUAGES`` in ``cbo/settings.py``. Notification e-mails are written in the language of the person who confirms the
breakfast.
