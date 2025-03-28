"""titiler.xarray.io"""

import os
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import attr
import boto3
import numpy
import logging
import s3fs
import xarray
from morecantile import TileMatrixSet
from rio_tiler.constants import WEB_MERCATOR_TMS
from rio_tiler.io.xarray import XarrayReader
from xarray.namedarray.utils import module_available

from titiler.core.auth import is_file_in_public_workspace, is_whitelisted_url

AWS_PRIVATE_ROLE_ARN = os.getenv("AWS_PRIVATE_ROLE_ARN")


def get_s3_filesystem(src_path: str, auth_token: Optional[str] = None) -> s3fs.S3FileSystem:
    """Return appropriate S3FileSystem based on authorization and URL type."""
    if auth_token and is_whitelisted_url(src_path) and not is_file_in_public_workspace(src_path):
        sts = boto3.client("sts")
        creds = sts.assume_role_with_web_identity(
            RoleArn=AWS_PRIVATE_ROLE_ARN,
            RoleSessionName="titiler-core",
            DurationSeconds=900,
            WebIdentityToken=auth_token,
        )["Credentials"]
        return s3fs.S3FileSystem(
            key=creds["AccessKeyId"],
            secret=creds["SecretAccessKey"],
            token=creds["SessionToken"],
        )

    elif is_file_in_public_workspace(src_path):
        session = boto3.Session()
        creds = session.get_credentials().get_frozen_credentials()
        return s3fs.S3FileSystem(
            key=creds.access_key,
            secret=creds.secret_key,
            token=creds.token,
            anon=False
        )

    return s3fs.S3FileSystem(anon=True)


def xarray_open_dataset(
    src_path: str,
    group: Optional[str] = None,
    decode_times: bool = True,
    request_options: Optional[Dict] = None,
) -> xarray.Dataset:
    """Open Xarray dataset from various sources (NetCDF, Zarr, Kerchunk)."""

    import fsspec  # noqa

    try:
        import h5netcdf
    except ImportError:  # pragma: nocover
        logging.warning("h5netcdf not installed")

    try:
        import zarr
    except ImportError:  # pragma: nocover
        logging.warning("zarr not installed")

    parsed = urlparse(src_path)
    protocol = parsed.scheme or "file"
    auth_token = (
        request_options.get("Authorization", "")
        if request_options and "Authorization" in request_options
        else ""
    )
    auth_token = auth_token.removeprefix("Bearer ")

    xr_open_args: Dict[str, Any] = {
        "decode_coords": "all",
        "decode_times": decode_times,
    }
    if group:
        xr_open_args["group"] = group

    # Kerchunk datasets (JSON references)
    if src_path.lower().endswith(".json"):
        fs = fsspec.filesystem("reference", fo=src_path).get_mapper("")
        return xarray.open_zarr(fs, **xr_open_args)

    # NetCDF files
    elif any(src_path.lower().endswith(ext) for ext in [".nc", ".nc4"]):
        fs = fsspec.filesystem(protocol)
        xr_open_args.update({"engine": "h5netcdf", "lock": False})
        with fs.open(src_path) as file:
            return xarray.open_dataset(file, **xr_open_args)

    # Zarr datasets
    else:
        xr_open_args["engine"] = "zarr"

        if protocol == "file":
            fs = fsspec.filesystem("file")
        else:
            fs = get_s3_filesystem(src_path, auth_token)

        mapper = fs.get_mapper(src_path)
        ds = xarray.open_dataset(mapper, consolidated=False, **xr_open_args)

        if not ds.keys():
            raise ValueError(f"Unable to open dataset or dataset is empty: {src_path}")

        return ds


def _arrange_dims(da: xarray.DataArray) -> xarray.DataArray:
    """Arrange coordinates and time dimensions.

    An rioxarray.exceptions.InvalidDimensionOrder error is raised if the coordinates are not in the correct order time, y, and x.
    See: https://github.com/corteva/rioxarray/discussions/674

    We conform to using x and y as the spatial dimension names..

    """
    if "x" not in da.dims and "y" not in da.dims:
        try:
            latitude_var_name = next(
                name
                for name in ["lat", "latitude", "LAT", "LATITUDE", "Lat"]
                if name in da.dims
            )
            longitude_var_name = next(
                name
                for name in ["lon", "longitude", "LON", "LONGITUDE", "Lon"]
                if name in da.dims
            )
        except StopIteration as e:
            raise ValueError(f"Couldn't find X/Y dimensions in {da.dims}") from e

        da = da.rename({latitude_var_name: "y", longitude_var_name: "x"})

    if "TIME" in da.dims:
        da = da.rename({"TIME": "time"})

    if extra_dims := [d for d in da.dims if d not in ["x", "y"]]:
        da = da.transpose(*extra_dims, "y", "x")
    else:
        da = da.transpose("y", "x")

    # If min/max values are stored in `valid_range` we add them in `valid_min/valid_max`
    vmin, vmax = da.attrs.get("valid_min"), da.attrs.get("valid_max")
    if "valid_range" in da.attrs and not (vmin is not None and vmax is not None):
        valid_range = da.attrs.get("valid_range")
        da.attrs.update({"valid_min": valid_range[0], "valid_max": valid_range[1]})

    return da


def get_variable(
    ds: xarray.Dataset,
    variable: str,
    datetime: Optional[str] = None,
    drop_dim: Optional[str] = None,
) -> xarray.DataArray:
    """Get Xarray variable as DataArray.

    Args:
        ds (xarray.Dataset): Xarray Dataset.
        variable (str): Variable to extract from the Dataset.
        datetime (str, optional): datetime to select from the DataArray.
        drop_dim (str, optional): DataArray dimension to drop in form of `{dimension}={value}`.

    Returns:
        xarray.DataArray: 2D or 3D DataArray.

    """
    da = ds[variable]

    if drop_dim:
        dim_to_drop, dim_val = drop_dim.split("=")
        da = da.sel({dim_to_drop: dim_val}).drop_vars(dim_to_drop)

    da = _arrange_dims(da)

    # Make sure we have a valid CRS
    crs = da.rio.crs or "epsg:4326"
    da = da.rio.write_crs(crs)

    if crs == "epsg:4326" and (da.x > 180).any():
        # Adjust the longitude coordinates to the -180 to 180 range
        da = da.assign_coords(x=(da.x + 180) % 360 - 180)

        # Sort the dataset by the updated longitude coordinates
        da = da.sortby(da.x)

    # TODO: Technically we don't have to select the first time, rio-tiler should handle 3D dataset
    if "time" in da.dims:
        if datetime:
            # TODO: handle time interval
            time_as_str = datetime.split("T")[0]
            if da["time"].dtype == "O":
                da["time"] = da["time"].astype("datetime64[ns]")

            da = da.sel(
                time=numpy.array(time_as_str, dtype=numpy.datetime64), method="nearest"
            )
        else:
            da = da.isel(time=0)

    assert len(da.dims) in [2, 3], "titiler.xarray can only work with 2D or 3D dataset"

    return da


@attr.s
class Reader(XarrayReader):
    """Reader: Open Zarr file and access DataArray."""

    src_path: str = attr.ib()
    variable: str = attr.ib()

    request_options = attr.ib(default=None)

    # xarray.Dataset options
    opener: Callable[..., xarray.Dataset] = attr.ib(default=xarray_open_dataset)

    group: Optional[str] = attr.ib(default=None)
    decode_times: bool = attr.ib(default=True)

    # xarray.DataArray options
    datetime: Optional[str] = attr.ib(default=None)
    drop_dim: Optional[str] = attr.ib(default=None)

    tms: TileMatrixSet = attr.ib(default=WEB_MERCATOR_TMS)

    ds: xarray.Dataset = attr.ib(init=False)
    input: xarray.DataArray = attr.ib(init=False)

    _dims: List = attr.ib(init=False, factory=list)

    def __attrs_post_init__(self):
        """Set bounds and CRS."""
        self.ds = self.opener(
            self.src_path,
            group=self.group,
            decode_times=self.decode_times,
            request_options=self.request_options,
        )

        self.input = get_variable(
            self.ds,
            self.variable,
            datetime=self.datetime,
            drop_dim=self.drop_dim,
        )
        super().__attrs_post_init__()

    def close(self):
        """Close xarray dataset."""
        self.ds.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """Support using with Context Managers."""
        self.close()
