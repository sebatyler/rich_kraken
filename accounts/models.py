from model_utils.models import TimeStampedModel

from django.contrib.auth.models import AbstractUser
from django.db import models


class User(TimeStampedModel, AbstractUser):
    firebase_uid = models.CharField(max_length=128, unique=True, null=True)
    profile_picture = models.URLField(max_length=512, null=True, blank=True)
