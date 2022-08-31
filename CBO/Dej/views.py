"""
views module
"""
# ================================================= Import des packages ================================================
# OS
import datetime
import functools
import json

# Data Vis
import bokeh
# Django
import django
from bokeh.resources import INLINE  # noqa
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse_lazy
from django.utils.safestring import mark_safe
from django.views.generic.base import TemplateView
from django.views.generic.edit import CreateView, DeleteView, UpdateView
from django.views.generic.list import ListView

# Homemade packages
from Dej.models import PetitDej, PetitDejForm, User
from Dej.models import ResponsablePetitDej
from Dej.utils import Calendar


# ====================================== Variable init =================================================================
LOGIN_ = '/login/'
RESS2 = bokeh.resources.Resources('cdn')


# ====================================== Definition functions ==========================================================
@functools.lru_cache(maxsize=256)
def get_possible_labels_users(label):
    results = [x[label] for x in User.objects.order_by().values(label).distinct()]
    results = sorted(results, key=lambda x: x.lower())
    return results


def participate(user, petit_dej):
    return (user in petit_dej.organisateurs) or (user in petit_dej.participants)


def arg_is_yes(x):
    return x in ["true", "True", True, "y", "Y", "o", "Oui", "oui", "yes", "YES"]


# ====================================== Classes def   =================================================================
class IndexView(LoginRequiredMixin, TemplateView):
    login_url = LOGIN_
    redirect_field_name = 'redirect_to'
    template_name = "home.html"

    def __init__(self, *args, **kwargs):
        super(TemplateView, self).__init__(*args, **kwargs)
        self.context = {}

    def get(self, request, *args, **kwargs):
        self.context["csrf_token"] = django.middleware.csrf.get_token(request)
        return render(request, 'home.html', {})


class PetitDejCreate(CreateView):
    model = PetitDej
    form_class = PetitDejForm
    template_name = "PetitDejCreate.html"


class PetitDejUpdate(UpdateView):
    model = PetitDej
    form_class = PetitDejForm
    template_name = "PetitDejUpdate.html"


class PetitDejDelete(DeleteView):
    model = PetitDej
    success_url = reverse_lazy('author-list')
    form_class = PetitDejForm
    template_name = "PetitDejDelete.html"


class ListUserView(LoginRequiredMixin, ListView):
    login_url = LOGIN_
    redirect_field_name = 'redirect_to'
    template_name = "UserList.html"
    model = User
    context_object_name = 'users'  # Default: object_list
    paginate_by = 200
    queryset = User.objects.all().order_by("-nombre_points")  # Default: Model.objects.all()

    def get_queryset(self):
        """Return the last five published questions."""
        queryset = self.queryset

        equipe_filter = self.request.GET.get("equipe_filter", None)

        if isinstance(equipe_filter, str) and (len(equipe_filter) == 0):
            equipe_filter = None

        return queryset.order_by('-nombre_points')

    def get_context_data(self, **kwargs):
        context = super(ListUserView, self).get_context_data(**kwargs)
        context["all_equipes"] = get_possible_labels_users("equipe")
        return context


class UserDetailView(LoginRequiredMixin, TemplateView):
    login_url = LOGIN_
    redirect_field_name = 'redirect_to'
    template_name = "UserDetailView.html"

    def get(self, request, email=None, *args, **kwargs):
        user = User.objects.get(email=email)
        petit_dejs = PetitDej.objects.all()
        all_points = [round(petit_dej.points().get(user.email, 0), 2) for petit_dej in petit_dejs]
        petit_dejs = [petit_dej.to_dict(point=point) for petit_dej, point in zip(petit_dejs, all_points)]
        return render(request, self.template_name, {"user": user,
                                                    "petit_dejs": petit_dejs})


class NewPetitDejView(TemplateView, LoginRequiredMixin):
    login_url = LOGIN_
    redirect_field_name = 'redirect_to'
    template_name = "PetitDejNew.html"

    def __init__(self, *args, **kwargs):
        self.context = {"css_files": RESS2.css_files,
                        "js_files": RESS2.js_files,
                        "bokeh_scripts": []}
        self.init_context()
        super(NewPetitDejView, self).__init__(*args, **kwargs)

    def init_context(self):
        today = datetime.date.today()
        cal = Calendar(today.year, today.month)
        html_cal = cal.formatmonth(theyear=None, themonth=None, withyear=True)
        self.context["calendar"] = mark_safe(html_cal)  # nosec

    def get(self, request, *args, **kwargs):
        self.context["csrf_token"] = django.middleware.csrf.get_token(request)
        return render(request, self.template_name, self.context)

    def post(self, request, *args, **kwargs):
        self.context["csrf_token"] = django.middleware.csrf.get_token(request)
        return self.get(request, *args, **kwargs)


class NewPetitDejViewVue(TemplateView):

    def post(self, request):
        data = {}

        # Recuperation et cleaning des parametres
        date = request.POST.get("date")
        if date is None or len(date) == 0:
            date = datetime.datetime.now()
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f')

        data["date"] = date.strftime('%Y-%m-%dT%H:%M:%S.%f')

        add_organisateur = arg_is_yes(request.POST.get("add_organisateur", False))
        add_designated_orga = arg_is_yes(request.POST.get("add_designated_orga", False))
        save = arg_is_yes(request.POST.get("save", False))

        # Mise à jour du petit dejeuner
        if "top_init" in request.POST:
            petit_dej = PetitDej(date=datetime.datetime.today())
            petit_dej.save()
        else:
            uri = request.POST["uri"]

            petit_dej = PetitDej.objects.get(uri=uri)
            petit_dej.date = date
            petit_dej.participation_validation = arg_is_yes(request.POST.get("participation_vali"))
            petit_dej.save()

            # Ajout des absents
            petit_dej.absents.clear()
            absents = User.objects.filter(email__in=json.loads(request.POST.get("absent_emails")))
            absents_email = [x.email for x in absents]
            petit_dej.absents.add(*list(absents))

            # Ajout des respos
            petit_dej.respos.all().delete()
            organisateurs = json.loads(request.POST.get("organisateurs"))
            organisateurs = [x for x in organisateurs if not x["delete"]]
            organisateurs = [x for x in organisateurs if (x["responsable"]["email"] not in absents_email)]

            for organisateur in organisateurs:
                responsable = ResponsablePetitDej(
                    responsable=User.objects.get(email=organisateur["responsable"]["email"]),
                    petit_dej=petit_dej,
                    fait_maison=organisateur["fait_maison"])
                responsable.save()

            # Ajout d'un nouvel organisateur
            if add_organisateur:
                petit_dej.tirage_aleatoire()
                data["add_organisateur"] = True

            if add_designated_orga:
                designated_orgna = request.POST["designated_orgna"]

                respo = ResponsablePetitDej(responsable=User.objects.get(email=designated_orgna),
                                            petit_dej=petit_dej,
                                            fait_maison=False)
                respo.save()
                data["add_designated_orga"] = True

            if save:
                petit_dej.send_mail()

        data["organisateurs"] = [x.to_dict() for x in petit_dej.respos.all()]
        data["absents"] = [x.email for x in petit_dej.absents.all()]
        data["possible_users"] = [user.to_dict() for user in petit_dej.possible_users]
        data["uri"] = petit_dej.uri

        return JsonResponse(data)
