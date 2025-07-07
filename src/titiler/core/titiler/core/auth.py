import logging
import os
import re
import urllib.parse
from typing import Optional, Tuple
from urllib.parse import urlparse

import boto3
import jwt
from fastapi import HTTPException, Request
from rasterio.session import AWSSession

CREDENTIALS_ENDPOINT = os.getenv("CREDENTIALS_ENDPOINT", "")
AWS_PRIVATE_ROLE_ARN = os.getenv("AWS_PRIVATE_ROLE_ARN")
DEFAULT_REGION = os.getenv("AWS_REGION", "eu-west-2")

EODH_WORKSPACE_URL_PATTERN = re.compile(
    os.getenv(
        "EODH_WORKSPACE_URL_PATTERN",
        r"^https://(?P<subdomain>[\w-]+)\.(?P<env>[\w-]+)\.eodatahub-workspaces\.org\.uk/files/(?P<bucket>workspaces-eodhp-[\w-]+)/(?P<key>.+)$",
    )
)
EODH_S3_HTTPS_PATTERN = re.compile(
    os.getenv(
        "EODH_S3_HTTPS_PATTERN",
        r"^https://(?P<bucket>workspaces-eodhp-[\w-]+)\.s3\.eu-west-2\.amazonaws\.com/(?P<workspace>[\w-]+)/(?P<key>.+)$",
    )
)
EODH_S3_BUCKET_PATTERN = re.compile(
    os.getenv("EODH_S3_BUCKET_PATTERN", r"^s3://workspaces-eodhp-[\w-]+/")
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


def is_whitelisted_url(url: str) -> bool:
    """
    Check if the given URL matches ANY of the patterns in WHITELIST_PATTERNS.
    Whitelisted URLs might require special S3 credentials.
    """
    return any(p.match(url) for p in WHITELIST_PATTERNS)


def is_file_in_public_workspace(src_path: str) -> bool:
    """
    Determines if the URL points to a file in a public workspace.
    In this context, a path is considered public if '/public/' is
    the second segment in the URL path.
    """
    if is_whitelisted_url(src_path):
        parts = urllib.parse.urlparse(src_path).path.split("/")
        return len(parts) > 2 and parts[2] == "public"
    return False


def rewrite_https_to_s3_if_needed(url: str) -> Tuple[str, Optional[str]]:
    """
    Converts certain HTTPS URLs to S3 URLs if they match known workspace patterns.
    Returns the rewritten S3 path and the subdomain or workspace name, if applicable.
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
    Forces conversion of an HTTPS URL to S3 format if it matches
    a general Amazon S3 HTTPS pattern, else returns the original URL.
    """
    if match := GENERAL_HTTPS_PATTERN.match(url):
        return f"s3://{match['bucket']}/{match['key']}"
    return url


def assume_aws_role_with_token(token: str) -> boto3.Session:
    """
    Uses a web identity token to assume the AWS private role,
    returning a boto3 Session configured with temporary credentials.
    """
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


def auth_token_in_request_header(request_headers: dict) -> bool:
    """
    Checks if the request contains an 'Authorization' header with a Bearer token.
    Returns True if the token is present, False otherwise.
    """
    auth_header = request_headers.get("Authorization", "")
    return bool(auth_header and auth_header.startswith("Bearer "))


def resolve_src_path_and_credentials(
    src_path: str,
    request: Request,
    gdal_env: dict,
) -> Tuple[str, dict]:
    """
    Resolves the given source path and determines the correct AWS credentials (if any).
    This covers four scenarios:
      1) A whitelisted S3/HTTPS URL (requires an AWS role assumption if not public).
      2) A public workspace URL (uses service account credentials).
      3) A non-DataHub URL (passed through as-is).
      4) A local EFS path (checks workspace authorisation via JWT claims).

    Returns:
        A tuple of (resolved_path, updated_gdal_environment).
    """
    updated_env = dict(gdal_env)
    parsed = urlparse(src_path)

    # 1) Whitelisted and auth header present (use AWS role assumption)
    if is_whitelisted_url(src_path) and auth_token_in_request_header(request.headers):
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
        updated_env["session"] = aws_session

    # 2) Whitelisted, in public workspace segment and no auth headers supplied (use service account credentials)
    elif is_file_in_public_workspace(src_path) and not auth_token_in_request_header(
        request.headers
    ):
        resolved_path, _ = rewrite_https_to_s3_if_needed(src_path)
        boto_session = boto3.Session()
        aws_session = AWSSession(boto_session, requester_pays=False)
        updated_env["session"] = aws_session

    # 3) Non-DataHub URL (S3/HTTPS/etc.)
    elif parsed.scheme and parsed.netloc:
        resolved_path = src_path

    # 4) Local EFS path (check JWT and normalise path)
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
    """
    Extracts the workspace name and relative path from an EFS path,
    ensuring it matches the expected format or raises an HTTPException.
    """
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
    """
    Checks if the given JWT claims permit access to the specified workspace.
    """
    return workspace in claims.get("workspaces", [])


def decode_jwt_token(auth_header: str) -> dict:
    """
    Decodes a JWT token from the 'Authorization' header without verifying the signature.
    Raises an HTTPException if the token is missing or invalid.
    """
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
