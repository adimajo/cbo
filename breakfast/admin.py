from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _

from breakfast.models import Breakfast, GlobalPointAdjustment, Host, User, UserPointAdjustment


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ("username", "last_name", "email", "team", "arrival_date", "departure_date", "specialty", "points")
    list_filter = ("team", "is_active", "is_staff")
    fieldsets = (
        *BaseUserAdmin.fieldsets,
        (_("Breakfast"), {"fields": ("team", "specialty", "bio", "arrival_date", "departure_date", "points")}),
    )
    add_fieldsets = (
        *BaseUserAdmin.add_fieldsets,
        (_("Breakfast"), {"fields": ("email", "team", "arrival_date")}),
    )
    readonly_fields = ("points",)


class HostInline(admin.TabularInline):
    model = Host
    extra = 0


@admin.register(Breakfast)
class BreakfastAdmin(admin.ModelAdmin):
    readonly_fields = ("token", "confirmed_at")
    list_display = (
        "date",
        "confirmed_at",
        "include_optional_team",
        "absentee_count",
        "participant_count",
        "host_count",
    )
    list_filter = ("include_optional_team",)
    inlines = [HostInline]


@admin.register(Host)
class HostAdmin(admin.ModelAdmin):
    list_display = ("user", "breakfast", "homemade")
    list_filter = ("homemade",)


@admin.register(UserPointAdjustment)
class UserPointAdjustmentAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "points")
    list_filter = ("user",)


@admin.register(GlobalPointAdjustment)
class GlobalPointAdjustmentAdmin(admin.ModelAdmin):
    list_display = ("points", "created_at")
