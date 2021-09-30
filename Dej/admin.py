# -*- coding: utf-8 -*-
#============================ Import statements============================================
from django.contrib import admin
import os, sys

# Gestion des exports Excel
from import_export import resources
from import_export.admin import ImportExportModelAdmin

from .models import User, UserForm
from .models import PetitDej, PetitDejForm
from .models import ResponsablePetitDej, ResponsablePetitDejForm
from .models import DeltaPointUser, DeltaPointUserForm
from .models import DeltaPoint, DeltaPointForm

class ResponsablePetitDejAdmin(admin.TabularInline):
    model = ResponsablePetitDej

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('username', "last_name", "email", "equipe", "date", "date_depart", "date_dernier_petit_dej", "specialite", "nombre_points")
    list_filter = ("equipe",)
    form = UserForm

@admin.register(PetitDej)
class PetitDejAdmin(admin.ModelAdmin):
    readonly_fields = ["uri",]
    list_display = ("date", "mois", "participation_validation",
                    "nombre_absents", "nombre_participants", "nombre_organisateurs")
    list_filter = ("participation_validation",)
    form = PetitDejForm
    inlines = [ResponsablePetitDejAdmin]


@admin.register(ResponsablePetitDej)
class ResponsablePetitDejAdmin(admin.ModelAdmin):
    list_display = ("responsable", "petit_dej", "fait_maison")
    list_filter = ("responsable", "petit_dej", "fait_maison")
    form = ResponsablePetitDejForm


@admin.register(DeltaPointUser)
class DeltaPointUserAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "points")
    list_filter = ("user", "created_at", "points")
    form = DeltaPointUserForm


@admin.register(DeltaPoint)
class DeltaPointAdmin(admin.ModelAdmin):
    list_display = ("points", "created_at",)
    list_filter = ("points", "created_at")
    form = DeltaPointForm
