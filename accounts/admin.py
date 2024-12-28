from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _

from .models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ("id", "email", "username", "is_staff", "firebase_uid", "created", "modified")
    list_filter = ("is_active", "is_staff", "is_superuser", "groups")
    search_fields = ("username", "email", "firebase_uid")
    list_display_links = ("id", "email")

    # 사용자 추가 비활성화
    add_form = None
    add_fieldsets = None

    fieldsets = (
        (None, {"fields": ("username",)}),
        (_("Personal info"), {"fields": ("email", "profile_picture")}),
        (_("Firebase info"), {"fields": ("firebase_uid",)}),
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                ),
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )

    def has_add_permission(self, request):
        """관리자 페이지에서 사용자 추가를 비활성화"""
        return False
