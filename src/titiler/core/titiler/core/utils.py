"""titiler.core utilities."""

import os
import re
import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import boto3
import numpy
import requests
from fastapi import HTTPException, Request
from rasterio.dtypes import dtype_ranges
from rasterio.session import AWSSession
from rio_tiler.colormap import apply_cmap
from rio_tiler.errors import InvalidDatatypeWarning
from rio_tiler.models import ImageData
from rio_tiler.types import ColorMapType, IntervalTuple
from rio_tiler.utils import linear_rescale, render

from titiler.core.resources.enums import ImageType

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


def rescale_array(
    array: numpy.ndarray,
    mask: numpy.ndarray,
    in_range: Sequence[IntervalTuple],
    out_range: Sequence[IntervalTuple] = ((0, 255),),
    out_dtype: Union[str, numpy.number] = "uint8",
) -> numpy.ndarray:
    """Rescale data array"""
    if len(array.shape) < 3:
        array = numpy.expand_dims(array, axis=0)

    nbands = array.shape[0]
    if len(in_range) != nbands:
        in_range = ((in_range[0]),) * nbands

    if len(out_range) != nbands:
        out_range = ((out_range[0]),) * nbands

    for bdx in range(nbands):
        array[bdx] = numpy.where(
            mask[bdx],
            linear_rescale(
                array[bdx], in_range=in_range[bdx], out_range=out_range[bdx]
            ),
            0,
        )

    return array.astype(out_dtype)


def render_image(
    image: ImageData,
    output_format: Optional[ImageType] = None,
    colormap: Optional[ColorMapType] = None,
    add_mask: bool = True,
    **kwargs: Any,
) -> Tuple[bytes, str]:
    """convert image data to file.

    This is adapted from https://github.com/cogeotiff/rio-tiler/blob/066878704f841a332a53027b74f7e0a97f10f4b2/rio_tiler/models.py#L698-L764
    """
    data, mask = image.data.copy(), image.mask.copy()
    datatype_range = image.dataset_statistics or (dtype_ranges[str(data.dtype)],)

    if colormap:
        data, alpha_from_cmap = apply_cmap(data, colormap)
        # Combine both Mask from dataset and Alpha band from Colormap
        mask = numpy.bitwise_and(alpha_from_cmap, mask)
        datatype_range = (dtype_ranges[str(data.dtype)],)

    # If output_format is not set, we choose between JPEG and PNG
    if not output_format:
        output_format = ImageType.jpeg if mask.all() else ImageType.png

    if output_format == ImageType.png and data.dtype not in ["uint8", "uint16"]:
        warnings.warn(
            f"Invalid type: `{data.dtype}` for the `{output_format}` driver. Data will be rescaled using min/max type bounds or dataset_statistics.",
            InvalidDatatypeWarning,
        )
        data = rescale_array(data, mask, in_range=datatype_range)

    elif output_format in [
        ImageType.jpeg,
        ImageType.jpg,
        ImageType.webp,
    ] and data.dtype not in ["uint8"]:
        warnings.warn(
            f"Invalid type: `{data.dtype}` for the `{output_format}` driver. Data will be rescaled using min/max type bounds or dataset_statistics.",
            InvalidDatatypeWarning,
        )
        data = rescale_array(data, mask, in_range=datatype_range)

    elif output_format == ImageType.jp2 and data.dtype not in [
        "uint8",
        "int16",
        "uint16",
    ]:
        warnings.warn(
            f"Invalid type: `{data.dtype}` for the `{output_format}` driver. Data will be rescaled using min/max type bounds or dataset_statistics.",
            InvalidDatatypeWarning,
        )
        data = rescale_array(data, mask, in_range=datatype_range)

    creation_options = {**kwargs, **output_format.profile}
    if output_format == ImageType.tif:
        if "transform" not in creation_options:
            creation_options.update({"transform": image.transform})
        if "crs" not in creation_options and image.crs:
            creation_options.update({"crs": image.crs})

    if not add_mask:
        mask = None

    return (
        render(
            data,
            mask,
            img_format=output_format.driver,
            **creation_options,
        ),
        output_format.mediatype,
    )


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


def resolve_src_path_and_credentials(
    src_path: str,
    request: Request,
    gdal_env: dict,
) -> tuple[str, dict]:

    updated_env = dict(gdal_env)

    # 1. Check if URL is in our "whitelist" (either https:// or s3:// for workspaces-eodhp-*)
    if is_whitelisted_url(src_path):
        resolved_path, workspace = rewrite_https_to_s3_if_needed(src_path)

        # s3 = boto3.client("s3")
        # creds = s3.assume_role_with_web_identity(
        #     RoleArn=AWS_ROLE_ARN,
        #     RoleSessionName="titiler-core",
        #     WebIdentityToken=request.headers.get("Authorization"),
        #     DurationSeconds=60,
        # )
        token = request.headers.get("Authorization").removeprefix("Bearer ")
        print('Token', token)

        sts = boto3.client("sts")
        creds = sts.assume_role_with_web_identity(
            RoleArn=AWS_ROLE_ARN,
            RoleSessionName="titiler-core",
            DurationSeconds=900,
            WebIdentityToken=token,
        )["Credentials"]

        print("Creds: ", creds)

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
        print("Updated environment with AWS session")

    else:
        resolved_path = src_path

    return resolved_path, updated_env
