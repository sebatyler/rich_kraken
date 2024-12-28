from .base import *
import sentry_sdk
from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration
from sentry_sdk.integrations.django import DjangoIntegration

DEBUG = False

# storage
DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
STATICFILES_STORAGE = "storages.backends.s3boto3.S3StaticStorage"
AWS_LOCATION = "static"
AWS_STORAGE_BUCKET_NAME = "rich-kraken-storage"
AWS_S3_REGION_NAME = "ap-northeast-2"
AWS_S3_CUSTOM_DOMAIN = "d33asjp2ow8o94.cloudfront.net"


sentry_sdk.init(
    dsn="https://2c3e7b1053eaee35c241d49173c12760@o262905.ingest.us.sentry.io/4507236051582976",
    integrations=[DjangoIntegration(), AwsLambdaIntegration()],
    send_default_pii=True,
    traces_sample_rate=0.1,
    profiles_sample_rate=0.1,
    environment=ENV,
)
