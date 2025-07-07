import jwt
import pytest

from titiler.core import auth


def test_is_whitelisted_url():
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
    public_url = "https://workspaces-eodhp-dev.s3.eu-west-2.amazonaws.com/my-workspace/public/myfile.tif"
    private_url = "https://workspaces-eodhp-dev.s3.eu-west-2.amazonaws.com/my-workspace/private/myfile.tif"
    assert auth.is_file_in_public_workspace(public_url)
    assert not auth.is_file_in_public_workspace(private_url)


def test_rewrite_https_to_s3_if_needed():
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
    url = "https://mybucket.s3.eu-west-1.amazonaws.com/path/file.tif"
    assert auth.rewrite_https_to_s3_force(url) == "s3://mybucket/path/file.tif"


def test_parse_efs_path():
    path = "/mnt/efs/workspace/path/file.txt"
    assert auth.parse_efs_path(path) == ("workspace", "path/file.txt")

    path2 = "workspace/path/file.txt"
    assert auth.parse_efs_path(path2) == ("workspace", "path/file.txt")


def test_is_workspace_authorized():
    claims = {"workspaces": ["myworkspace"]}
    assert auth.is_workspace_authorized("myworkspace", claims)
    assert not auth.is_workspace_authorized("otherworkspace", claims)


def test_decode_jwt_token():
    token = jwt.encode({"workspaces": ["test"]}, "secret", algorithm="HS256")
    decoded = auth.decode_jwt_token(f"Bearer {token}")
    assert decoded["workspaces"] == ["test"]

    with pytest.raises(auth.HTTPException) as excinfo:
        auth.decode_jwt_token("wrongformat")
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "Missing authorization token"
