"""Basic auth string formatting checks"""

import types
from unittest import mock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from titiler.core import auth

PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(key: rsa.RSAPrivateKey, aud: str = "account", **claims: object) -> str:
    return jwt.encode(
        {"sub": "test-user", "aud": aud, **claims}, key, algorithm="RS256"
    )


@pytest.fixture(autouse=True)
def mock_jwks():
    """Stand in for a call out to Keycloak's JWKS endpoint."""
    with mock.patch.object(auth, "_jwks_client") as mock_client:
        mock_client.return_value.get_signing_key_from_jwt.return_value = (
            types.SimpleNamespace(key=PRIVATE_KEY.public_key())
        )
        yield mock_client


def test_is_whitelisted_url():
    """Checks our URL whitelist is working as it should."""
    assert auth.is_whitelisted_url(
        "https://workspaces-eodhp-dev.s3.eu-west-2.amazonaws.com/mydata/file.tif"
    )
    assert auth.is_whitelisted_url("s3://workspaces-eodhp-staging/mydata/file.tif")
    assert auth.is_whitelisted_url(
        "https://sub.env.eodatahub-workspaces.org.uk/files/workspaces-eodhp-prod/file.tif"
    )
    assert auth.is_whitelisted_url(
        "https://long-workspace-name.prod.eodatahub-workspaces.org.uk/files/workspaces-eodhp-newenv/file.tif"
    )
    assert not auth.is_whitelisted_url("https://example.com/notallowed/file.tif")


def test_is_file_in_public_workspace():
    """See if we can spot files in a public workspace."""
    public_url = "https://workspaces-eodhp-dev.s3.eu-west-2.amazonaws.com/my-workspace/public/myfile.tif"
    private_url = "https://workspaces-eodhp-dev.s3.eu-west-2.amazonaws.com/my-workspace/private/myfile.tif"
    assert auth.is_file_in_public_workspace(public_url)
    assert not auth.is_file_in_public_workspace(private_url)


def test_rewrite_https_to_s3_if_needed():
    """Make sure we're rewriting HTTPS URLs to S3 correctly."""
    url1 = "https://user.env.eodatahub-workspaces.org.uk/files/workspaces-eodhp-dev/myfile.tif"
    assert auth.rewrite_https_to_s3_if_needed(url1) == (
        "s3://workspaces-eodhp-dev/user/myfile.tif",
        "user",
    )

    url2 = (
        "https://workspaces-eodhp-staging.s3.eu-west-2.amazonaws.com/workspace/file.tif"
    )
    assert auth.rewrite_https_to_s3_if_needed(url2) == (
        "s3://workspaces-eodhp-staging/workspace/file.tif",
        "workspace",
    )

    url3 = "https://unmatched.example.com/file.tif"
    assert auth.rewrite_https_to_s3_if_needed(url3) == (url3, None)


def test_rewrite_https_to_s3_force():
    """Check we can always switch an S3 https URL to an S3 URI."""
    url = "https://mybucket.s3.eu-west-1.amazonaws.com/path/file.tif"
    assert auth.rewrite_https_to_s3_force(url) == "s3://mybucket/path/file.tif"


def test_parse_efs_path():
    """Make sure we can break down an EFS path properly."""
    path = "/mnt/efs/workspace/path/file.txt"
    assert auth.parse_efs_path(path) == ("workspace", "path/file.txt")

    path2 = "workspace/path/file.txt"
    assert auth.parse_efs_path(path2) == ("workspace", "path/file.txt")


def test_is_workspace_authorized():
    """Check if a user has access to a given workspace."""
    claims = {"workspaces": ["myworkspace"]}
    assert auth.is_workspace_authorized("myworkspace", claims)
    assert not auth.is_workspace_authorized("otherworkspace", claims)


def test_decode_jwt_token():
    """Make sure our JWT token decoding works and throws errors when it should."""
    token = _token(PRIVATE_KEY, workspaces=["test"])
    decoded = auth.decode_jwt_token(f"Bearer {token}")
    assert decoded["workspaces"] == ["test"]

    with pytest.raises(auth.HTTPException) as excinfo:
        auth.decode_jwt_token("wrongformat")
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "Missing authorization token"


def test_decode_jwt_token_forged_signature_is_rejected():
    """This is the exact bug that shipped: verify_signature was False, so any signature -
    including one that is not cryptographically valid at all - was accepted.
    """
    header = jwt.utils.base64url_encode(b'{"alg":"RS256","typ":"JWT"}').decode()
    payload = jwt.utils.base64url_encode(
        b'{"sub":"attacker","workspaces":["someone-elses-workspace"],"aud":"account"}'
    ).decode()
    forged_signature = jwt.utils.base64url_encode(b"not-a-real-signature").decode()
    forged_token = f"{header}.{payload}.{forged_signature}"

    with pytest.raises(auth.HTTPException) as excinfo:
        auth.decode_jwt_token(f"Bearer {forged_token}")
    assert excinfo.value.status_code == 403


def test_decode_jwt_token_signed_by_a_different_key_is_rejected():
    """A token signed by any key other than Keycloak's should be rejected."""
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _token(other_key)

    with pytest.raises(auth.HTTPException) as excinfo:
        auth.decode_jwt_token(f"Bearer {token}")
    assert excinfo.value.status_code == 403


def test_decode_jwt_token_wrong_audience_is_rejected():
    """A validly signed token issued for a different client should be rejected."""
    token = _token(PRIVATE_KEY, aud="some-other-client")

    with pytest.raises(auth.HTTPException) as excinfo:
        auth.decode_jwt_token(f"Bearer {token}")
    assert excinfo.value.status_code == 403
