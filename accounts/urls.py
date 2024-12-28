from django.contrib.auth.views import LogoutView
from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("verify-token/", views.verify_firebase_token, name="verify_token"),
    path("logout/", LogoutView.as_view(next_page="/"), name="logout"),
]
