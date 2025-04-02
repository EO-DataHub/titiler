import os
import jwt
import logging
import re
from typing import Tuple, Optional
import urllib.parse
from urllib.parse import urlparse

import boto3
from fastapi import Request
from rasterio.session import AWSSession

from fastapi import HTTPException

CREDENTIALS_ENDPOINT = os.getenv("CREDENTIALS_ENDPOINT", "")
AWS_PRIVATE_ROLE_ARN = os.getenv("AWS_PRIVATE_ROLE_ARN")
DEFAULT_REGION = os.getenv("AWS_REGION", "eu-west-2")

EODH_WORKSPACE_URL_PATTERN = re.compile(
    os.getenv(
        "EODH_WORKSPACE_URL_PATTERN",
        ""
    )
)
EODH_S3_HTTPS_PATTERN = re.compile(
    os.getenv(
        "EODH_S3_HTTPS_PATTERN",
        ""
    )
)
EODH_S3_BUCKET_PATTERN = re.compile(
    os.getenv(
        "EODH_S3_BUCKET_PATTERN",
        r"^s3://workspaces-eodhp-[\w-]+/"
    )
)

WHITELIST_PATTERNS = [
    EODH_WORKSPACE_URL_PATTERN,
    EODH_S3_HTTPS_PATTERN,
    EODH_S3_BUCKET_PATTERN,
]


EFS_PATH_PATTERN = re.compile(r"^/mnt/efs/(?P<workspace>[^/]+)/(?P<path>.+)$")

GENERAL_HTTPS_PATTERN = re.compile(
    r"^https://(?P<bucket>[\w-]+)\.s3\.[\w-]+\.amazonaws\.com/(?P<key>.+)$"
)


def force_no_irsa() -> None:
    """
    Remove environment variables that cause botocore/boto3 to attempt
    STS AssumeRoleWithWebIdentity for IRSA (EKS). This ensures we rely
    ONLY on the ephemeral credentials we fetch below.
    """
    for var in ("AWS_ROLE_ARN", "AWS_WEB_IDENTITY_TOKEN_FILE"):
        os.environ.pop(var, None)


def is_whitelisted_url(url: str) -> bool:
    """
    Check if the given URL matches ANY of the patterns in WHITELIST_PATTERNS.
    If yes, we want to attach ephemeral S3 credentials.
    """
    return any(p.match(url) for p in WHITELIST_PATTERNS)


def is_file_in_public_workspace(src_path: str) -> bool:
    """
    Check if the given URL is in a public
    workspace (i.e. /public/ as the second index in the path).
    """
    if is_whitelisted_url(src_path):
        parts = urllib.parse.urlparse(src_path).path.split("/")
        return len(parts) > 2 and parts[2] == "public"
    return False


def rewrite_https_to_s3_if_needed(url: str) -> Tuple[str, Optional[str]]:
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
    if match := EODH_WORKSPACE_URL_PATTERN.match(url):
        return (
            f"s3://{match['bucket']}/{match['subdomain']}/{match['key']}",
            match["subdomain"],
        )
    if match := EODH_S3_HTTPS_PATTERN.match(url):
        return (
            f"s3://{match['bucket']}/{match['workspace']}/{match['key']}",
            match["workspace"],
        )
    return url, None


def rewrite_https_to_s3_force(url: str) -> str:
    """
    Rewrite HTTPS URLs to S3 URLs for any pattern.

    Args:
        url: The HTTPS URL to be rewritten

    Returns:
        String of (s3_url)
    """
    if match := GENERAL_HTTPS_PATTERN.match(url):
        return f"s3://{match['bucket']}/{match['key']}"
    return url


def assume_aws_role_with_token(token: str) -> boto3.Session:
    """Assume AWS role using web identity token; returns AWS Session."""
    sts = boto3.client("sts")
    creds = sts.assume_role_with_web_identity(
        RoleArn=AWS_PRIVATE_ROLE_ARN,
        RoleSessionName="titiler-core",
        DurationSeconds=900,
        WebIdentityToken=token,
    )["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=DEFAULT_REGION,
    )


def resolve_src_path_and_credentials(
    src_path: str,
    request: Request,
    gdal_env: dict,
) -> Tuple[str, dict]:
    """Resolve path and AWS credentials depending on source URL and authorization."""
    updated_env = dict(gdal_env)
    parsed = urlparse(src_path)

    if is_whitelisted_url(src_path) and not is_file_in_public_workspace(src_path):
        resolved_path, _ = rewrite_https_to_s3_if_needed(src_path)
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=403,
                detail="Missing authorization token",
            )

        token = auth_header.removeprefix("Bearer ")
        boto_session = assume_aws_role_with_token(token)
        aws_session = AWSSession(boto_session, requester_pays=False)
        updated_env.pop("AWS_ACCESS_KEY_ID", None)
        updated_env.pop("AWS_SECRET_ACCESS_KEY", None)
        updated_env.pop("AWS_SESSION_TOKEN", None)
        updated_env["session"] = aws_session

    elif parsed.scheme and parsed.netloc:
        resolved_path = src_path

    else:
        workspace, path = parse_efs_path(src_path)
        claims = decode_jwt_token(request.headers.get("Authorization", ""))
        if not is_workspace_authorized(workspace, claims):
            raise HTTPException(
                status_code=403,
                detail="Unauthorized access to workspace",
            )
        resolved_path = os.path.normpath(f"/mnt/efs/{workspace}/{path}")

    return resolved_path, updated_env


def parse_efs_path(src_path: str) -> Tuple[str, str]:
    """Parse EFS path into workspace and relative path components."""
    if not src_path.startswith("/mnt/efs"):
        src_path = f"/mnt/efs/{src_path.lstrip('/')}"

    normalized_path = os.path.normpath(src_path)
    match = EFS_PATH_PATTERN.match(normalized_path)
    if match:
        return match.group("workspace"), match.group("path")
    raise HTTPException(
        status_code=400,
        detail="Invalid EFS path format",
    )


def is_workspace_authorized(workspace: str, claims: dict) -> bool:
    """Verify if the JWT claims authorize access to the workspace."""
    return workspace in claims.get("workspaces", [])


def decode_jwt_token(auth_header: str) -> dict:
    """Decode JWT token from Authorization header without signature verification."""
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=403,
            detail="Missing authorization token",
        )

    token_str = auth_header.removeprefix("Bearer ")
    try:
        return jwt.decode(token_str, options={"verify_signature": False})
    except jwt.DecodeError as e:
        logging.exception("Failed to decode JWT token")
        raise HTTPException(
            status_code=403,
            detail="Invalid token",
        ) from e
