"""
models module
"""
# -*- coding: utf-8 -*-
# ================================================= Import des packages ================================================
# Gestion des données
# Dates
import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from django.contrib.auth.models import AbstractUser
# Package Django
from django.db import models
from django.forms import ModelForm
from django.core.mail import EmailMultiAlternatives
from icalendar import Calendar, Event, Alarm
from icalendar import vCalAddress, vText
import tempfile
import os
from datetime import timedelta

# ============================================
first_last_name = "{} - {}"
admin_email = "adrien.ehrhardt@credit-agricole-sa.fr"
INFORMATIONS_G_N_RALES = "Informations générales"
LAST_UPDATE = "Last update"
CREATION_DATE = "Creation date"
NOMBRE_DE_POINTS = "Nombre de points"
EQUIPE_CHOICES = (("MIG", "MIG"),
                  ("Validation", "Validation"),
                  ("GRO", "GRO"),
                  ("MOD retail", "MOD retail"),
                  ("MOD corp", "MOD corp"),
                  ("IFRS9", "IFRS9"))


# ============================================
def _get_today():
    return datetime.date.today()


def _generate_unique_uri():
    """Generates a unique uri for the chat session."""
    return str(uuid4()).replace('-', '')[:15]


class User(AbstractUser):
    date = models.DateField("Date d'arrivée", default=_get_today)
    date_depart = models.DateField("Date de départ de MIG", blank=True, null=True)
    bio = models.TextField("Biographie", max_length=500, blank=True)
    specialite = models.CharField("Specialité", max_length=30, blank=True)
    nombre_points = models.IntegerField(NOMBRE_DE_POINTS, null=True, blank=True)
    equipe = models.CharField("Equipe", default="MIG", choices=EQUIPE_CHOICES, max_length=40)

    @property
    def date_dernier_petit_dej(self):
        if self.is_respo_of.all().count() > 0:
            return self.is_respo_of.order_by("petit_dej__date").first().petit_dej.date
        return "-"

    def to_dict(self, **kwargs):
        dict_result = {"username": self.username,
                       "first_name": self.first_name,
                       "last_name": self.last_name,
                       "email": self.email,
                       }
        dict_result.update(kwargs)
        return dict_result


class UserForm(ModelForm):
    class Meta:
        model = User
        exclude = []


class PetitDej(models.Model):
    created_at = models.DateTimeField(CREATION_DATE, auto_now_add=True)
    updated_at = models.DateTimeField(LAST_UPDATE, auto_now=True)
    date = models.DateTimeField("Date du petit dej")
    uri = models.URLField(default=_generate_unique_uri)

    participation_validation = models.BooleanField("Participation de la vali?", default=True)
    absents = models.ManyToManyField(User, related_name='petit_dej', blank=True)

    def update_all_points(self):
        all_users = User.objects.all()
        all_delta = sum([delta.points for delta in DeltaPoint.objects.all()])
        user_points = {x.email: all_delta for x in all_users}

        # Offset par utilisateur
        for delta_point_user in DeltaPointUser.objects.all():
            email = delta_point_user.user.email
            user_points[delta_point_user.user.email] += delta_point_user.points

        # Calcul des points pour chque personne pour chaque petit dejeuner
        for petit_dej in PetitDej.objects.exclude(uri=self.uri):
            point = petit_dej.points()
            for user in all_users:
                user_points[user.email] += point.get(user.email, 0)

        # Update the points
        for email, points in user_points.items():
            user = User.objects.get(email=email)
            if not user.nombre_points == points:
                user.nombre_points = points
                user.save()

    def __str__(self):
        return self.date.strftime(("%A, %d. %B %Y %I:%M%p"))

    def to_dict(self, **kwargs):
        results = {"id": self.id,
                   "pk": self.pk,
                   "date": self.date,
                   "uri": self.uri,
                   "nombre_absents": self.nombre_absents,
                   "nombre_participants": self.nombre_participants,
                   "nombre_organisateurs": self.nombre_organisateurs,

                   "participation_validation": self.participation_validation}
        results.update(kwargs)
        return results

    @property
    def mois(self):
        return self.date.month

    @property
    def nombre_absents(self):
        return self.absents.all().count()

    @property
    def possible_users(self):
        queryset = User.objects.filter(date__lte=self.date).exclude(date_depart__lte=self.date)
        if not self.participation_validation:
            queryset = User.objects.exclude(equipe="Validation")
        return queryset

    @property
    def participants(self):
        queryset = [x for x in self.possible_users if x not in self.organisateurs]
        queryset = [x for x in queryset if x not in self.absents.all()]
        return queryset

    @property
    def participants_point_negatifs(self):
        return [x for x in self.participants if x.nombre_point <= 0]

    @property
    def nombre_participants(self):
        return len(self.participants)

    @property
    def organisateurs(self):
        return [x.responsable for x in self.respos.all()]

    @property
    def nombre_organisateurs(self):
        return len(self.organisateurs)

    def get_responsable_list(self):
        d = "<ul>"
        for respo in ResponsablePetitDej.objects.filter(petit_dej=self):
            d += "<li>{}</li>".format(respo.responsable.username)
        d += "</ul>"
        return d

    def tirage_aleatoire(self):
        self.update_all_points()
        user_points = {x.email: x.nombre_points for x in self.participants}

        # Filtre sur les users d'interets
        min_points = min(list(
            user_points.values()))
        # Si tous le monde a des points positifs, on prend la personne ayant le moins de points.
        real_min_points = max(min_points, 0)
        records = [{"email": email, "points": points} for email, points in user_points.items() if
                   points <= real_min_points]

        if len(records) == 1:
            return records[0]["email"]

        # La probabilité d'etre tiré est affine en fonction du nombre de points.
        df = pd.DataFrame(records)
        df["p"] = real_min_points + 1 - df["points"]  # Je ramene tous le monde entre 1 et +infini
        df["p"] = df["p"] / df["p"].sum()

        # Tirage aleatoire
        prochain = np.random.choice(df["email"].tolist(), size=1, p=df["p"].tolist())
        email = prochain[0]

        respo = ResponsablePetitDej(responsable=User.objects.get(email=email),
                                    petit_dej=self,
                                    fait_maison=False)
        respo.save()

        return respo

    def points(self):
        organisateurs = self.organisateurs
        participants = self.participants

        participants_par_orga = len(participants) / len(organisateurs)

        result = {}
        for user in participants:
            result[user.email] = -1
        for respo in self.respos.all():
            result[respo.responsable.email] = participants_par_orga * (1 + 0.5 * respo.fait_maison)
        return result

    def save(self, *args, **kwargs):
        super(PetitDej, self).save(*args, **kwargs)

    def send_mail(self):
        # Recuperation des liste d'organisateurs et de participants
        organisateurs = self.organisateurs
        participants = self.participants
        nb_participants_par_orga = int(len(participants) / len(organisateurs))

        scores = [{"name": x.username, "points": x.nombre_points} for x in self.possible_users]
        scores = sorted(scores, key=lambda x: x["points"])

        premier_quart = scores[:int(len(scores) / 4)]
        dernier_quart = scores[int(3 * len(scores) / 4):]

        tableau_scores_up = "\n\t- ".join(["{}".format(x["name"]) for x in premier_quart])
        tableau_scores_down = "\n\t- ".join(["{}".format(x["name"]) for x in dernier_quart])

        date_str = self.date.strftime(("%d/%m/%Y à %I:%M%p"))
        subject = "Petit déjeuner du {}".format(date_str)

        body_organ = "Bonjour, " \
                     "\n\nVous avez été désignés comme organisateurs du petit-déjeuner qui aura lieu le {}.\n\n" \
                     "" \
                     "Le nombre de participants attendus par organisateur est de {}.\n\n" \
                     "" \
                     "Si vous avez la possibilité de préparer des plats maison, vous obtiendrez des points" \
                     "bonus!\n\n" \
                     "Bien cordialement\n\n" \
                     "" \
                     "CBO".format(date_str, nb_participants_par_orga)

        body_participant = "Bonjour, " \
                           "\n\nVous etes conviés au petit-déjeuner qui aura lieu le {}.\n\n" \
                           "" \
                           "Pour information, les organisateurs désignés sont :\n\t- {}\n\n" \
                           "Le tableau des scores devient:\n\n" \
                           "Peu de points : \n\t- {}\n\n" \
                           "Nombreux points : \n\t- {}\n\n" \
                           "CBO".format(date_str,
                                        "\n\t- ".join([x.username for x in organisateurs]),
                                        tableau_scores_up,
                                        tableau_scores_down)

        # 1) Mail pour les responsables
        mail = EmailMultiAlternatives(subject=subject,
                                      body=body_organ,
                                      from_email="Groupe-recherche-operationnelle.GRO@credit-agricole-sa.fr",
                                      to=[x.email for x in organisateurs],
                                      cc=[admin_email],
                                      reply_to=[admin_email])

        # Générer le fichier .ical
        temp = tempfile.NamedTemporaryFile()
        with open(temp.name + ".ics", 'wb') as f:
            cal = Calendar()
            event = Event()
            event.add('summary', subject)
            event.add('dtstart', self.date)
            event.add('dtend', self.date + timedelta(seconds=1800))

            organizer = vCalAddress('MAILTO:nicolas.damay@credit-agricole-sa.fr')
            organizer.params['cn'] = vText('Nicolas DAMAY')
            organizer.params['role'] = vText('CHAIR')
            event['organizer'] = organizer
            event['location'] = vText('Forum')
            event.add('priority', 5)

            for participant in organisateurs:
                attendee = vCalAddress('MAILTO:{}'.format(participant.email))
                attendee.params['cn'] = vText(first_last_name.format(participant.first_name, participant.last_name))
                attendee.params['ROLE'] = vText('REQ-PARTICIPANT')
                event.add('attendee', attendee, encode=0)

            for participant in participants:
                attendee = vCalAddress('MAILTO:{}'.format(participant.email))
                attendee.params['cn'] = vText(first_last_name.format(participant.first_name, participant.last_name))
                attendee.params['ROLE'] = vText('OPT-PARTICIPANT')
                event.add('attendee', attendee, encode=0)

            alarm = Alarm()
            alarm.add('trigger', timedelta(days=-2))
            alarm.add('action', 'display')
            alarm.add('description', "Plus que deux jours pour préparer des petits plats")
            event.add_component(alarm)

            alarm = Alarm()
            alarm.add('trigger', timedelta(days=-1))
            alarm.add('action', 'display')
            alarm.add('description', "Le petit déjeuner aura lieu demain")
            event.add_component(alarm)

            cal.add_component(event)
            f.write(cal.to_ical())
        mail.attach_file(temp.name + ".ics")
        mail.send()
        os.remove(temp.name + ".ics")

        # 2) Mail pour les participants
        mail = EmailMultiAlternatives(subject=subject,
                                      body=body_participant,
                                      from_email="Groupe-recherche-operationnelle.GRO@credit-agricole-sa.fr",
                                      to=[x.email for x in participants],
                                      cc=[admin_email],
                                      reply_to=[admin_email])

        # Générer le fichier .ical
        temp = tempfile.NamedTemporaryFile()
        with open(temp.name + ".ics", 'wb') as f:
            cal = Calendar()
            event = Event()
            event.add('summary', subject)
            event.add('dtstart', self.date)
            event.add('dtend', self.date + timedelta(seconds=1800))

            organizer = vCalAddress('MAILTO:nicolas.damay@credit-agricole-sa.fr')
            organizer.params['cn'] = vText('Nicolas DAMAY')
            organizer.params['role'] = vText('CHAIR')
            event['organizer'] = organizer
            event['location'] = vText('Forum')
            event.add('priority', 5)

            for participant in organisateurs:
                attendee = vCalAddress('MAILTO:{}'.format(participant.email))
                attendee.params['cn'] = vText(first_last_name.format(participant.first_name, participant.last_name))
                attendee.params['ROLE'] = vText('REQ-PARTICIPANT')
                event.add('attendee', attendee, encode=0)

            for participant in participants:
                attendee = vCalAddress('MAILTO:{}'.format(participant.email))
                attendee.params['cn'] = vText(first_last_name.format(participant.first_name, participant.last_name))
                attendee.params['ROLE'] = vText('OPT-PARTICIPANT')
                event.add('attendee', attendee, encode=0)
            cal.add_component(event)
            f.write(cal.to_ical())
        mail.attach_file(temp.name + ".ics")
        mail.send()
        os.remove(temp.name + ".ics")


class PetitDejForm(ModelForm):
    class Meta:
        model = PetitDej
        exclude = []

    layout = {INFORMATIONS_G_N_RALES: ["date", "participation_validation"],
              "Absents": ["absents"]}


class ResponsablePetitDej(models.Model):
    created_at = models.DateTimeField(CREATION_DATE, auto_now_add=True)
    updated_at = models.DateTimeField(LAST_UPDATE, auto_now=True)
    responsable = models.ForeignKey(User, on_delete=models.CASCADE, related_name="is_respo_of")
    petit_dej = models.ForeignKey(PetitDej, on_delete=models.CASCADE, related_name="respos")
    fait_maison = models.BooleanField("Fait maison?", default=False)

    def to_dict(self):
        return {"responsable": self.responsable.to_dict(),
                "delete": False,
                "fait_maison": self.fait_maison}

    def save(self, *args, **kwargs):
        super(ResponsablePetitDej, self).save(*args, **kwargs)


class ResponsablePetitDejForm(ModelForm):
    class Meta:
        model = ResponsablePetitDej
        exclude = ["already_saved"]

    layout = {INFORMATIONS_G_N_RALES: ["responsable", "petit_dej"],
              "Bonus": ["fait_maison"]}


class DeltaPointUser(models.Model):
    created_at = models.DateTimeField(CREATION_DATE, auto_now_add=True)
    updated_at = models.DateTimeField(LAST_UPDATE, auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    points = models.IntegerField(NOMBRE_DE_POINTS, default=0)


class DeltaPointUserForm(ModelForm):
    class Meta:
        model = DeltaPointUser
        exclude = []

    layout = {INFORMATIONS_G_N_RALES: ["user", "points"],
              }


class DeltaPoint(models.Model):
    created_at = models.DateTimeField(CREATION_DATE, auto_now_add=True)
    updated_at = models.DateTimeField(LAST_UPDATE, auto_now=True)
    points = models.IntegerField(NOMBRE_DE_POINTS, default=0)


class DeltaPointForm(ModelForm):
    class Meta:
        model = DeltaPoint
        exclude = []

    layout = {INFORMATIONS_G_N_RALES: ["points"]}
