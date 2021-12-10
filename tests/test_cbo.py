import os
import sys

import django

sys.path.append("CBO")

if os.environ.get('DJANGO_SETTINGS_MODULE') is None:
    os.environ['DJANGO_SETTINGS_MODULE'] = 'CBO.DjangoSite.settings'


def test_cbo():
    django.setup()
