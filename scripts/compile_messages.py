"""Compile every ``.po`` file under ``breakfast/locale`` to ``.mo`` (a pure-Python stand-in for ``msgfmt``).

Usage: ``uv run python scripts/compile_messages.py``. Contributors with GNU gettext installed can use
``python manage.py makemessages -l <code>`` and ``python manage.py compilemessages`` instead.
"""

from pathlib import Path

import polib

LOCALE_DIR = Path(__file__).resolve().parent.parent / "breakfast" / "locale"


def compile_messages() -> list[Path]:
    compiled = []
    for po_path in sorted(LOCALE_DIR.glob("*/LC_MESSAGES/*.po")):
        mo_path = po_path.with_suffix(".mo")
        polib.pofile(str(po_path)).save_as_mofile(str(mo_path))
        compiled.append(mo_path)
    return compiled


if __name__ == "__main__":
    for path in compile_messages():
        print(f"compiled {path.relative_to(LOCALE_DIR.parent.parent)}")
