import logging

from firebase_admin import auth

from django.conf import settings
from django.contrib.auth import login
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .models import User


@csrf_exempt
@require_POST
def verify_firebase_token(request):
    try:
        data = request.POST
        token = data.get("token")
        next_url = request.GET.get("next", "/")

        decoded_token = auth.verify_id_token(token)

        user, _ = User.objects.get_or_create(
            firebase_uid=decoded_token["uid"],
            defaults={
                "username": decoded_token.get("name", ""),
                "email": decoded_token.get("email", ""),
                "profile_picture": decoded_token.get("picture", ""),
            },
        )

        login(request, user)

        return HttpResponse(
            f"""
            <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
                로그인 성공! 리다이렉트 중...
            </div>
            <script>
                setTimeout(() => {{
                    window.location.href = '{next_url}';
                }}, 1000);
            </script>
            """
        )

    except Exception as e:
        logging.warning(f"Verify Firebase Token Error: {e=}")
        return HttpResponseBadRequest(
            """
            <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                인증 처리 중 오류가 발생했습니다.
            </div>
            """,
        )


def admin_login(request):
    next_url = request.GET.get("next", reverse("admin:index"))
    return render(
        request,
        "admin_login.html",
        {"next_url": next_url, "firebase_config": settings.FIREBASE_CONFIG},
    )
