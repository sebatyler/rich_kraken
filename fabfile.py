import json
from os import environ

import dotenv
import requests
from fabric import task
from invoke import run as local

dotenv.read_dotenv()

PROJECT = "rich_trader"
PROFILE = "sebatyler"

IS_CI = environ.get("GITHUB_ACTIONS") == "true"

target = "prod"
built_images = set()

DEFAULT_REGION = "ap-northeast-2"
AWS_COMMAND = f"aws --region {DEFAULT_REGION}"

access_key, secret_key = [
    local(f"{AWS_COMMAND} --profile {PROFILE} configure get {key}", echo=False, hide=True).stdout.strip()
    for key in ("aws_access_key_id", "aws_secret_access_key")
]

environ.update(
    # docker pip caching
    DOCKER_BUILDKIT="1",
    # skip checking python version
    ZAPPA_RUNNING_IN_DOCKER="True",
    AWS_ACCESS_KEY_ID=access_key,
    AWS_SECRET_ACCESS_KEY=secret_key,
)


@task
def nb(c):
    """
    no build - skip build
    """
    global built_images
    built_images = None


def _get_docker_image():
    return f"{PROJECT}-{target}"


@task
def build(c, use_cache=True):
    """
    build docker image for zappa
    """
    global built_images
    image = _get_docker_image()
    package_file = "requirements.txt"

    if built_images is not None and image not in built_images:
        # generate zappa_settings.py
        local(f"zappa save-python-settings-file {target}")

        # docker build
        progress = environ.get("DOCKER_PROGRESS")  # possible value: plain
        cmd_list = [
            "docker buildx build --platform linux/amd64 --provenance false",
            "-f deploy/Dockerfile",
            f"--build-arg USE_CACHE={1 if use_cache else 0}",
            f"--build-arg PACKAGE_FILE={package_file}",
            f"--build-arg TARGET={target}",
            f"--build-arg aws_access_key_id={access_key}",
            f"--build-arg aws_secret_access_key={secret_key}",
            f"-t {_get_docker_image()}:latest",
        ]
        if progress:
            cmd_list.append(f"--progress={progress}")

        cmd_list.append(".")
        local(" ".join(cmd_list))
        built_images.add(image)


@task
def run(c, command=""):
    """
    run command in docker image
    """
    local(f"docker run -it --rm --entrypoint bash {_get_docker_image()}:latest {command}")


def _zappa_run(operation=None, args=""):
    if not operation:
        raise ValueError("operation required")

    command = f"zappa {operation} {target}"

    if args:
        command += f' "{args}"'

    local(command, echo=True)


@task
def tail(c):
    """
    tail logs
    """
    _zappa_run("tail")


@task
def status(c):
    """
    show status
    """
    _zappa_run("status")


@task
def setup(c):
    """
    first deploy to AWS lambda & API gateway
    """
    _deploy_with_ecr(c, update=False)


@task
def deploy(c):
    """
    deploy to AWS lambda & API gateway
    """

    _deploy_with_ecr(c, update=True)


@task
def remove_zappa_files(c):
    """
    remove files generated during zappa deployment
    """
    local(f"rm -fv {PROJECT}-*.tar.gz")
    local(f"rm -fv {PROJECT}-*template*.json")


@task
def rollback(c, n=1):
    """
    rollback to previous deployment
    """
    _zappa_run("rollback", f"-n {n}")


def _run_manage_command(command, with_zappa=None, add_noinput=True):
    args = command
    if add_noinput:
        args += " --noinput"

    if with_zappa:
        _zappa_run("manage", args)
    else:
        key = "DJANGO_SETTINGS_MODULE"
        settings = environ.get(key)

        environ[key] = f"rich_trader.settings.{target}"
        local(f"python manage.py {args}", echo=True)

        if settings:
            environ[key] = settings
        else:
            del environ[key]


@task
def collectstatic(c):
    """
    run collectstatic command to upload static files to S3
    """
    _run_manage_command("collectstatic", with_zappa=True)


@task
def schedule(c):
    """
    schedule functions
    """
    _zappa_run("schedule")


@task
def unschedule(c):
    """
    unschedule functions
    """
    _zappa_run("unschedule")


@task
def migrate(c, with_zappa=True):
    """
    run migrate command to apply DB migrations
    """
    if target == "prod" and not IS_CI:
        raise EnvironmentError("migration for prod must be run only in Github Actions")

    if with_zappa:
        _zappa_run("manage", args="migrate --noinput")
    else:
        _run_manage_command("migrate")


@task
def github_actions_deploy(c):
    """
    deploy via Github Actions
    """
    if not IS_CI:
        raise EnvironmentError("Environment is not Github Actions")

    # deployment
    _deploy_with_ecr(c, update=True, use_cache=False)

    collectstatic(c)


@task
def create_db(c):
    """
    create DB
    """
    _run_manage_command("create_db", with_zappa=True, add_noinput=False)


# Set custom domain name at API Gateway
@task
def setup_ssl(c):
    """
    setup SSL Certification
    """
    _zappa_run("certify")


@task
def destroy(c):
    """
    destroy
    """
    if target in ("prod", "staging"):
        raise ValueError("You can't destroy it when target is prod or staging.")

    _zappa_run("undeploy")


def _create_ecr_repo(c):

    repo_name = _get_docker_image()
    aws_default = f"{AWS_COMMAND} --output json ecr"
    local(
        f"{aws_default} create-repository --repository-name {repo_name} --image-scanning-configuration scanOnPush=true"
    )

    set_ecr_lifecycle(c, force=False)


def _deploy_with_ecr(c, update=True, use_cache=True):
    # https://ianwhitestone.work/zappa-serverless-docker/
    # https://docs.aws.amazon.com/lambda/latest/dg/images-create.html
    build(c, use_cache=use_cache)

    aws_default = f"{AWS_COMMAND} --output json ecr"
    image_name = _get_docker_image()

    # get repository url
    for try_create in (True, False):
        result = local(f"{aws_default} describe-repositories --repository-names {image_name}", warn=True)

        if result:
            break

        if try_create:
            _create_ecr_repo(c)

    result_dict = json.loads(result.stdout)
    repo_url = result_dict["repositories"][0]["repositoryUri"]

    # re-tag
    local(f"docker tag {image_name}:latest {repo_url}:latest")

    # get authenticated to push to ECR
    local(f"{aws_default} get-login-password | docker login --username AWS --password-stdin {repo_url}")
    # push it
    local(f"docker push {repo_url}:latest")

    # deploy (first time) or update
    command = "update" if update else "deploy"
    local(f"zappa {command} {target} -d {repo_url}:latest", echo=True)


@task
def set_ecr_lifecycle(c, force=True):
    """set Elastic Container Registry lifecycle"""
    repo_name = _get_docker_image()
    aws_default = f"{AWS_COMMAND} --output json ecr"

    result = local(f"{aws_default} get-lifecycle-policy --repository-name {repo_name}", warn=True)

    if force or result.exited != 0:
        local(
            f"{aws_default} put-lifecycle-policy --repository-name {repo_name} --lifecycle-policy-text file://./deploy/ecr_lifecycle_policy.json"
        )


@task
def run_server(c, use_cache=True):
    """
    Run server in docker
    """
    build(c, use_cache=use_cache)
    local(f"docker run -p 9000:8080 --env-file=.env {_get_docker_image()}:latest")


@task
def test_request(c, path="/"):
    """
    Test request to server in docker
    """
    # https://ianwhitestone.work/zappa-serverless-docker/
    # https://docs.aws.amazon.com/lambda/latest/dg/images-test.html

    res = requests.post(
        "http://localhost:9000/2015-03-31/functions/function/invocations",
        json=dict(httpMethod="GET", requestContext={}, body=None, path=path),
    )
    print(res.json())
