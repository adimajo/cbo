
import sys
import os
import django
sys.path.append(r"N:\Projets02\SiteMRU\2019 - PetitDejs\DjangoSite")
os.environ['DJANGO_SETTINGS_MODULE'] = 'DjangoSite.settings'
django.setup()


from Dej.models import PetitDej, User

petit_dej = PetitDej.objects.all()[0]

users = User.objects.filter(date__lte=petit_dej.date)
if not petit_dej.participation_validation:
    users = users.exclude(equipe="Validation")
users = users.difference(petit_dej.absents.all())
organisateurs = [x.responsable for x in petit_dej.respos.all()]
participants = [x for x in users if not x in organisateurs]

participants_par_orga = len(participants) / len(organisateurs)

result = {}
for user in participants:
    result[user.email] = -1
for respo in petit_dej.respos.all():
    result[respo.responsable.email] = participants_par_orga * (1 + 0.2 * respo.fait_maison)

