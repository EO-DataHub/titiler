import os
import jwt
import logging
import re
import urllib.parse
from urllib.parse import urlparse

import boto3
from fastapi import Request
from rasterio.session import AWSSession

CREDENTIALS_ENDPOINT = os.getenv(
    "CREDENTIALS_ENDPOINT", "https://staging.eodatahub.org.uk/api/workspaces/workspace-name/me/s3-tokens"
)
AWS_PRIVATE_ROLE_ARN = os.getenv("AWS_PRIVATE_ROLE_ARN")
AWS_PUBLIC_ROLE_ARN = os.getenv("AWS_PUBLIC_ROLE_ARN")

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
        if re.match(pattern, url):
            return True
    return False


def is_file_in_public_workspace(src_path: str) -> bool:
    """
    Check if the given URL is in a public
    workspace (i.e. /public/ as the second index in the path).
    """
    if is_whitelisted_url(src_path):
        parts = urllib.parse.urlparse(src_path).path.split('/')
        print(parts)
        if len(parts) > 2 and parts[2] == 'public':
            return True


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


def rewrite_https_to_s3_force(url: str) -> str:
    """
    Rewrite HTTPS URLs to S3 URLs for any pattern.

    Args:
        url: The HTTPS URL to be rewritten

    Returns:
        String of (s3_url)
    """
    https_pattern = (
        r"^https://([\w-]+)\.s3\.[\w-]+\.amazonaws\.com/(.+)$"
    )
    match = re.match(https_pattern, url)

    if match:
        bucket = match.group(1)
        key = match.group(2)
        return f"s3://{bucket}/{key}"

    return url


def resolve_src_path_and_credentials(
    src_path: str,
    request: Request,
    gdal_env: dict,
) -> tuple[str, dict]:

    updated_env = dict(gdal_env)
    parsed = urlparse(src_path)

    # 1. Check if URL is in our "whitelist" (either https:// or s3:// for workspaces-eodhp-*)
    if is_whitelisted_url(src_path) and not is_file_in_public_workspace(src_path):
        resolved_path, workspace = rewrite_https_to_s3_if_needed(src_path)
        token = request.headers.get("Authorization", "")
        if not token:
            raise PermissionError("Missing authorization token. The Workspace API key is either missing or invalid.")
        token = token.removeprefix("Bearer ")

        sts = boto3.client("sts")
        creds = sts.assume_role_with_web_identity(
            RoleArn=AWS_PRIVATE_ROLE_ARN,
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

    elif parsed.scheme != '' and parsed.netloc != '':
        resolved_path = src_path
    else:
        resolved_path = parse_file_path_and_check_access(src_path, request.headers)

    return resolved_path, updated_env


def parse_file_path_and_check_access(src_path: str, request_options: dict) -> str:
    """
    Parses the provided file path and verifies if the user is authorised to access the specified workspace.
    """

    if not src_path.startswith("/mnt/efs"):
        src_path = f"/mnt/efs/{src_path.lstrip('/')}"

    auth_header = request_options.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise PermissionError("Missing or invalid authorization token")

    token_str = auth_header.removeprefix("Bearer ")

    try:
        token = jwt.decode(token_str, options={"verify_signature": False})
    except Exception as e:
        logging.exception("Failed to decode JWT token")
        raise PermissionError("Invalid token") from e

    normalized_path = os.path.normpath(src_path)

    # Ensure the path is still within /mnt/efs
    base_dir = os.path.normpath("/mnt/efs")
    if not normalized_path.startswith(base_dir):
        logging.error("Path does not start with the expected base directory")
        raise ValueError("Invalid file path: must reside within /mnt/efs")

    # Extract the workspace segment from the path.
    path_parts = normalized_path.split(os.sep)
    if len(path_parts) < 4:
        logging.error("Invalid file path structure; cannot extract workspace")
        raise ValueError("Invalid file path structure")

    workspace = path_parts[3]

    # get allowed workspaces from token claims.
    allowed_workspaces = token.get("workspaces", [])
    allowed_member_groups = token.get("member_groups", [])

    # Verify that the workspace is authorized.
    if workspace not in allowed_workspaces and workspace not in allowed_member_groups:
        raise PermissionError(f"Access denied for workspace: {workspace}")

    return normalized_path
