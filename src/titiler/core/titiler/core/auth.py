import os
import re

import boto3
from fastapi import Request
from rasterio.session import AWSSession

CREDENTIALS_ENDPOINT = os.getenv(
    "CREDENTIALS_ENDPOINT", "https://staging.eodatahub.org.uk/api/workspaces/workspace-name/me/s3-tokens"
)
AWS_ROLE_ARN = os.getenv("AWS_ROLE_ARN")
DEFAULT_REGION = os.getenv("AWS_REGION", "eu-west-2")
WHITELIST_PATTERNS = [
    r"^https://workspaces-eodhp-[\w-]+\.s3\.eu-west-2\.amazonaws\.com/",
    r"^s3://workspaces-eodhp-[\w-]+/",
    r"^https://([\w-]+)\.([\w-]+)\.eodatahub-workspaces\.org\.uk/files/workspaces-eodhp-[\w-]+/",
]


def force_no_irsa() -> None:
    """
    Remove environment variables that cause botocore/boto3 to attempt
    STS AssumeRoleWithWebIdentity for IRSA (EKS). This ensures we rely
    ONLY on the ephemeral credentials we fetch below.
    """
    for var in ("AWS_ROLE_ARN", "AWS_WEB_IDENTITY_TOKEN_FILE"):
        if var in os.environ:
            del os.environ[var]


def is_whitelisted_url(url: str) -> bool:
    """
    Check if the given URL matches ANY of the patterns in WHITELIST_PATTERNS.
    If yes, we want to attach ephemeral S3 credentials.
    """
    for pattern in WHITELIST_PATTERNS:
        if "/public/" not in url and url.endswith('.tif') and re.match(pattern, url):
            return True
    return False


def rewrite_https_to_s3_if_needed(url: str):
    """
    Rewrite HTTPS URLs to S3 URLs when possible.
    For a URL of the form:
      https://workspaces-eodhp-<env>.s3.eu-west-2.amazonaws.com/<key>
    it returns:
      s3://workspaces-eodhp-<env>/<key>
    And for the new style:
      https://<subdomain>.<env>.eodatahub-workspaces.org.uk/files/workspaces-eodhp-<env>/<key>
    it returns:
      s3://workspaces-eodhp-<env>/<subdomain>/<key>
    """
    new_style_pattern = (
        r"^https://([\w-]+)\.([\w-]+)\.eodatahub-workspaces\.org\.uk/files/(workspaces-eodhp-[\w-]+)/(.+)$"
    )
    match = re.match(new_style_pattern, url)
    if match:
        workspace = match.group(1)       # e.g. "james-hinton"
        bucket_part = match.group(3)       # e.g. "workspaces-eodhp-staging"
        key_part = match.group(4)          # the remaining path after the bucket
        return (f"s3://{bucket_part}/{workspace}/{key_part}", workspace)

    # S3 URL pattern:
    https_pattern = (
        r"^https://(workspaces-eodhp-[\w-]+)\.s3\.eu-west-2\.amazonaws\.com/([\w-]+)/(.+)$"
    )
    match = re.match(https_pattern, url)
    if match:
        bucket_part = match.group(1)
        workspace = match.group(2)
        key_part = match.group(3)
        return (f"s3://{bucket_part}/{workspace}/{key_part}", workspace)

    return (url, None)


def rewrite_https_to_s3_force(url: str):
    """
    Rewrite HTTPS URLs to S3 URLs for any pattern.

    Args:
        url: The HTTPS URL to be rewritten

    Returns:
        Tuple of (s3_url, workspace)
    """
    https_pattern = (
        r"^https://([\w-]+)\.s3\.[\w-]+\.amazonaws\.com/(.+)$"
    )
    match = re.match(https_pattern, url)

    if match:
        bucket = match.group(1)
        key = match.group(2)
        return (f"s3://{bucket}/{key}", None)

    return (url, None)


def resolve_src_path_and_credentials(
    src_path: str,
    request: Request,
    gdal_env: dict,
) -> tuple[str, dict]:

    updated_env = dict(gdal_env)

    # 1. Check if URL is in our "whitelist" (either https:// or s3:// for workspaces-eodhp-*)
    if is_whitelisted_url(src_path):
        resolved_path, workspace = rewrite_https_to_s3_if_needed(src_path)
        token = request.headers.get("Authorization").removeprefix("Bearer ")

        sts = boto3.client("sts")
        creds = sts.assume_role_with_web_identity(
            RoleArn=AWS_ROLE_ARN,
            RoleSessionName="titiler-core",
            DurationSeconds=900,
            WebIdentityToken=token,
        )["Credentials"]

        # 5. Create a dedicated boto3 Session with ephemeral creds
        boto_session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=DEFAULT_REGION,
        )
        aws_session = AWSSession(boto_session, requester_pays=False)
        updated_env.pop("AWS_ACCESS_KEY_ID", None)
        updated_env.pop("AWS_SECRET_ACCESS_KEY", None)
        updated_env.pop("AWS_SESSION_TOKEN", None)
        updated_env["session"] = aws_session

    else:
        resolved_path = src_path

    return resolved_path, updated_env