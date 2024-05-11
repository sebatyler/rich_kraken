from .base import *
import sentry_sdk
from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration
from sentry_sdk.integrations.django import DjangoIntegration

DEBUG = False

sentry_sdk.init(
    dsn="https://2c3e7b1053eaee35c241d49173c12760@o262905.ingest.us.sentry.io/4507236051582976",
    integrations=[DjangoIntegration(), AwsLambdaIntegration()],
    send_default_pii=True,
    traces_sample_rate=0.1,
    profiles_sample_rate=0.1,
    environment=ENV,
)
