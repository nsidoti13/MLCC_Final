"""
src/data/download.py
====================
Download functions for each raw data source used in the wildfire ignition
prediction pipeline.

All functions follow the same contract:
  - Accept an output directory (defaulting to data/raw/<source>).
  - Use requests + tqdm for streaming downloads with progress bars.
  - Return a list of local file paths that were written.
  - Raise descriptive errors on HTTP or API failures.

Data sources and credentials
-----------------------------
MODIS     — NASA EarthData login via ~/.netrc or env vars
            EARTHDATA_USER / EARTHDATA_PASSWORD.
gridMET   — No authentication required.  University of Idaho THREDDS server.
NDFD      — No authentication required.  NOAA public FTP/HTTPS (operational only).
ERA5      — Copernicus CDS API key required.  Place in ~/.cdsapirc or set
            env vars CDSAPI_URL + CDSAPI_KEY.
            Register at https://cds.climate.copernicus.eu/user/register
NOAA CFSv2— No authentication required (legacy; gridMET preferred for history).
LANDFIRE  — No authentication required for public WCS endpoint.
Terrain   — No authentication required for TNM S3 bucket.
Human     — US Census API key via env var CENSUS_API_KEY.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _load_config() -> dict:
    """Load the master YAML config."""
    with open(_CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


def _ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stream_download(url: str, dest: Path, session: requests.Session, chunk_size: int = 1 << 20) -> Path:
    """
    Stream-download *url* into *dest*, showing a tqdm progress bar.

    Parameters
    ----------
    url:
        Remote URL to fetch.
    dest:
        Local file path to write.  Parent directory must already exist.
    session:
        A requests.Session (may carry auth headers).
    chunk_size:
        Bytes per chunk (default 1 MB).

    Returns
    -------
    Path
        The local path that was written.
    """
    if dest.exists():
        logger.info("Skipping %s — already downloaded.", dest.name)
        return dest

    logger.info("Downloading %s → %s", url, dest)
    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=dest.name,
        leave=False,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))

    return dest


# ---------------------------------------------------------------------------
# MODIS Fire (MOD14A1)
# ---------------------------------------------------------------------------


def download_modis_fire(
    start_year: int = 2015,
    end_year: int = 2023,
    output_dir: Optional[Path] = None,
    tiles: Optional[List[Tuple[int, int]]] = None,
) -> List[Path]:
    """
    Download MODIS Terra Thermal Anomalies/Fire Daily Global 1km (MOD14A1 v061).

    **Data source**: NASA EOSDIS LPDAAC — https://lpdaac.usgs.gov/products/mod14a1v061/

    MOD14A1 is an 8-day composite product, but we use the underlying daily
    fire-mask layer to derive point ignitions.  The HDF tiles covering
    California are h08v04 and h08v05.

    Authentication
    --------------
    NASA EarthData credentials must be available via one of:
      * ``~/.netrc``  (recommended for CI): ``machine urs.earthdata.nasa.gov login <u> password <p>``
      * Environment variables ``EARTHDATA_USER`` / ``EARTHDATA_PASSWORD``.

    Parameters
    ----------
    start_year:
        First year to download (inclusive).
    end_year:
        Last year to download (inclusive).
    output_dir:
        Destination directory.  Defaults to ``data/raw/modis/``.
    tiles:
        List of (h, v) MODIS tile indices.  Defaults to California tiles
        [(8, 4), (8, 5)].

    Returns
    -------
    List[Path]
        Paths to all downloaded HDF files.
    """
    config = _load_config()
    base_url = config["data"]["modis"]["base_url"]

    if output_dir is None:
        output_dir = Path(config["data"]["raw_dir"]) / "modis"
    output_dir = _ensure_dir(Path(output_dir))

    if tiles is None:
        tiles = [(8, 4), (8, 5)]  # California coverage

    # Build a session with optional EarthData auth
    session = requests.Session()
    earthdata_user = os.environ.get("EARTHDATA_USER")
    earthdata_pass = os.environ.get("EARTHDATA_PASSWORD")
    if earthdata_user and earthdata_pass:
        session.auth = (earthdata_user, earthdata_pass)
    else:
        logger.warning(
            "EARTHDATA_USER / EARTHDATA_PASSWORD not set.  "
            "Falling back to ~/.netrc for NASA EarthData authentication."
        )

    downloaded: List[Path] = []

    for year in range(start_year, end_year + 1):
        # MOD14A1 is a daily product; iterate over Julian days
        for doy in range(1, 366, 8):  # 8-day composites
            date_str = f"{year}{doy:03d}"
            folder_url = f"{base_url}{year}.{doy:03d}/"

            # Discover files in the directory listing
            try:
                resp = session.get(folder_url, timeout=30)
                resp.raise_for_status()
            except requests.HTTPError as exc:
                logger.warning("Could not list MODIS directory %s: %s", folder_url, exc)
                continue

            for h, v in tiles:
                tile_tag = f"h{h:02d}v{v:02d}"
                # Find the matching HDF filename in the HTML listing
                for line in resp.text.splitlines():
                    if tile_tag in line and ".hdf" in line and "xml" not in line:
                        # Extract filename from href
                        start = line.find('href="') + 6
                        end = line.find('"', start)
                        if start < 6 or end < 0:
                            continue
                        filename = line[start:end]
                        file_url = folder_url + filename
                        dest = output_dir / filename
                        try:
                            path = _stream_download(file_url, dest, session)
                            downloaded.append(path)
                        except requests.HTTPError as exc:
                            logger.error("Failed to download %s: %s", file_url, exc)
                        break

    logger.info("MODIS: downloaded %d files to %s", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# NOAA CFSv2 Weather Forecasts
# ---------------------------------------------------------------------------


def download_noaa_weather(
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    output_dir: Optional[Path] = None,
    variables: Optional[List[str]] = None,
) -> List[Path]:
    """
    Download NOAA Climate Forecast System v2 (CFSv2) daily forecast GRIB2 files.

    **Data source**: NOAA NCEI CFSv2 archive
    https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system

    CFSv2 provides 6-hourly forecasts out to 9 months.  We download the
    daily-mean surface fields at forecast horizons +24 h … +168 h to build
    7-day forecast feature vectors:

    Variables downloaded
    --------------------
    tmp2m   — 2-metre air temperature (K)
    rh2m    — 2-metre relative humidity (%)
    wnd10m  — 10-metre wind speed (m/s)
    apcp    — Accumulated precipitation (kg/m²)

    The files are global 0.5° GRIB2 grids; we later clip to California in the
    preprocessing stage.

    Parameters
    ----------
    start_date:
        ISO date string for first forecast initialization date.
    end_date:
        ISO date string for last forecast initialization date.
    output_dir:
        Destination directory.  Defaults to ``data/raw/noaa/``.
    variables:
        Override the variable list from config.

    Returns
    -------
    List[Path]
        Paths to all downloaded GRIB2 files.
    """
    import datetime

    config = _load_config()
    base_url = config["data"]["noaa_cfs"]["base_url"]
    if variables is None:
        variables = config["data"]["noaa_cfs"]["variables"]
    forecast_hours = config["data"]["noaa_cfs"]["forecast_hours"]

    if output_dir is None:
        output_dir = Path(config["data"]["raw_dir"]) / "noaa"
    output_dir = _ensure_dir(Path(output_dir))

    session = requests.Session()

    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    delta = datetime.timedelta(days=1)

    downloaded: List[Path] = []
    current = start

    while current <= end:
        ymd = current.strftime("%Y%m%d")
        year = current.strftime("%Y")
        month = current.strftime("%m")

        for var in variables:
            for fh in forecast_hours:
                # CFSv2 operational archive path pattern
                filename = f"cfs.{ymd}00.{var}.forecast.global.0p5.daily.grb2"
                file_url = f"{base_url}cfs.{ymd}/00/6hrly_grib_{var}/{filename}"
                dest = output_dir / year / month / filename
                _ensure_dir(dest.parent)

                try:
                    path = _stream_download(file_url, dest, session)
                    downloaded.append(path)
                except requests.HTTPError as exc:
                    logger.warning("NOAA: skipping %s (%s)", filename, exc)

        current += delta
        # Be polite to the server
        time.sleep(0.05)

    logger.info("NOAA: downloaded %d files to %s", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# LANDFIRE Vegetation and Fuels
# ---------------------------------------------------------------------------


def download_landfire(
    layers: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    bbox: Optional[dict] = None,
) -> List[Path]:
    """
    Download LANDFIRE 2022 (LF2022) raster layers for California via the
    LANDFIRE Product Service (LFPS) REST API.

    **Data source**: USGS LANDFIRE — https://www.landfire.gov/
    API docs: https://lfps.usgs.gov/helpdocs/productstable.html

    Layers downloaded (configurable via config.yaml)
    -------------------------------------------------
    US_220_EVT     — Existing Vegetation Type  (200+ classes)
    US_220_FBFM40  — Fire Behavior Fuel Model 40 (Scott & Burgan)
    US_220_CBD     — Canopy Bulk Density (kg/m³ × 100)
    US_220_CBH     — Canopy Base Height (m × 10)
    US_220_CC      — Canopy Cover (%)
    US_220_CH      — Canopy Height (m × 10)

    The API returns a job ID; we poll until the export is complete and then
    download the resulting GeoTIFF ZIP archive.

    Parameters
    ----------
    layers:
        Override the list of LANDFIRE layer codes from config.
    output_dir:
        Destination directory.  Defaults to ``data/raw/landfire/``.
    bbox:
        Bounding box dict with keys lon_min, lat_min, lon_max, lat_max.
        Defaults to California bbox from config.

    Returns
    -------
    List[Path]
        Paths to downloaded ZIP archives (each contains a GeoTIFF).
    """
    config = _load_config()
    base_url = config["data"]["landfire"]["base_url"]

    if layers is None:
        layers = config["data"]["landfire"]["layers"]
    if bbox is None:
        bbox = config["grid"]["bbox"]
    if output_dir is None:
        output_dir = Path(config["data"]["raw_dir"]) / "landfire"
    output_dir = _ensure_dir(Path(output_dir))

    session = requests.Session()
    downloaded: List[Path] = []

    for layer in tqdm(layers, desc="LANDFIRE layers"):
        # Build the LFPS job submission request
        params = {
            "layer": layer,
            "aoi": (
                f"{bbox['lon_min']},{bbox['lat_min']},"
                f"{bbox['lon_max']},{bbox['lat_max']}"
            ),
            "outputProjection": "4326",
            "resolutionUnits": "meter",
            "resolution": "30",
            "f": "json",
        }

        try:
            submit_resp = session.get(
                base_url + "submitJob",
                params=params,
                timeout=60,
            )
            submit_resp.raise_for_status()
        except requests.HTTPError as exc:
            logger.error("LANDFIRE submit failed for layer %s: %s", layer, exc)
            continue

        job_data = submit_resp.json()
        job_id = job_data.get("jobId")
        if not job_id:
            logger.error("LANDFIRE: no jobId returned for layer %s", layer)
            continue

        # Poll for job completion (max 10 minutes)
        status_url = base_url + f"jobs/{job_id}"
        max_polls = 60
        for poll_idx in range(max_polls):
            time.sleep(10)
            status_resp = session.get(status_url, timeout=30)
            status_resp.raise_for_status()
            status = status_resp.json().get("jobStatus", "")
            if status == "esriJobSucceeded":
                break
            if status in ("esriJobFailed", "esriJobCancelled"):
                logger.error("LANDFIRE job %s ended with status %s", job_id, status)
                break
        else:
            logger.warning("LANDFIRE job %s timed out after %d polls.", job_id, max_polls)
            continue

        # Retrieve download URL from job results
        results_resp = session.get(status_url + "/results/Output_File", timeout=30)
        results_resp.raise_for_status()
        download_url = results_resp.json().get("value", {}).get("url")
        if not download_url:
            logger.error("LANDFIRE: no download URL for layer %s job %s", layer, job_id)
            continue

        dest = output_dir / f"{layer}.zip"
        try:
            path = _stream_download(download_url, dest, session)
            downloaded.append(path)
        except requests.HTTPError as exc:
            logger.error("LANDFIRE download failed for layer %s: %s", layer, exc)

    logger.info("LANDFIRE: downloaded %d layer archives to %s", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# Terrain (USGS 3DEP 1/3 arc-second DEM)
# ---------------------------------------------------------------------------


def download_terrain(
    output_dir: Optional[Path] = None,
    bbox: Optional[dict] = None,
) -> List[Path]:
    """
    Download USGS 3DEP (3D Elevation Program) 1/3 arc-second DEM tiles
    covering California from The National Map S3 bucket.

    **Data source**: USGS The National Map — https://www.usgs.gov/the-national-map
    S3 bucket: s3://prd-tnm/StagedProducts/Elevation/13/TIFF/

    The 1/3 arc-second (~10 m) DEM is used to derive:
      - Elevation (m)
      - Slope (degrees)
      - Aspect (degrees)
    via GDAL in the preprocessing stage.

    Tiles are 1°×1° GeoTIFF files named ``USGS_13_<n><w>.tif``.
    For California we need tiles spanning lat 32–42 N, lon 114–125 W.

    Parameters
    ----------
    output_dir:
        Destination directory.  Defaults to ``data/raw/terrain/``.
    bbox:
        Bounding box dict.  Defaults to California bbox from config.

    Returns
    -------
    List[Path]
        Paths to all downloaded GeoTIFF DEM tiles.
    """
    config = _load_config()
    base_url = config["data"]["terrain"]["base_url"]

    if bbox is None:
        bbox = config["grid"]["bbox"]
    if output_dir is None:
        output_dir = Path(config["data"]["raw_dir"]) / "terrain"
    output_dir = _ensure_dir(Path(output_dir))

    session = requests.Session()
    downloaded: List[Path] = []

    # Enumerate 1×1 degree tiles covering the bounding box
    lat_min = int(bbox["lat_min"])
    lat_max = int(bbox["lat_max"]) + 1
    lon_min = int(abs(bbox["lon_max"])) - 1  # westernmost longitude (positive)
    lon_max = int(abs(bbox["lon_min"])) + 1

    tile_urls = []
    for lat in range(lat_min, lat_max):
        for lon in range(lon_min, lon_max):
            # TNM naming: n{lat+1}w{lon+1}, zero-padded to 2 digits
            tile_name = f"USGS_13_n{lat+1:02d}w{lon+1:03d}.tif"
            url = f"{base_url}n{lat+1:02d}w{lon+1:03d}/{tile_name}"
            tile_urls.append((url, tile_name))

    for url, tile_name in tqdm(tile_urls, desc="DEM tiles"):
        dest = output_dir / tile_name
        try:
            path = _stream_download(url, dest, session)
            downloaded.append(path)
        except requests.HTTPError as exc:
            logger.warning("Terrain tile not found (skipping): %s — %s", tile_name, exc)

    logger.info("Terrain: downloaded %d DEM tiles to %s", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# Human infrastructure (roads, powerlines, population)
# ---------------------------------------------------------------------------


def download_human_features(
    output_dir: Optional[Path] = None,
    state_fips: str = "06",  # California FIPS code
) -> List[Path]:
    """
    Download human-infrastructure layers used as ignition-risk covariates.

    **Data sources**:

    1. Roads — TIGER/Line 2023 shapefile (US Census Bureau)
       https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
       All road segments for California county-level files are fetched and
       merged.  Road density (km of road per km²) is computed per H3 cell.

    2. Electric Power Transmission Lines — HIFLD Open Data
       https://hifld-geoplatform.opendata.arcgis.com/datasets/electric-power-transmission-lines
       High-voltage transmission lines are a significant ignition risk factor.
       Distance to nearest line is computed per H3 cell.

    3. Population Density — US Census Bureau ACS 5-year estimates (Table B01003)
       Fetched via the Census Data API.  Requires env var ``CENSUS_API_KEY``.

    Parameters
    ----------
    output_dir:
        Destination directory.  Defaults to ``data/raw/human/``.
    state_fips:
        State FIPS code.  Default ``"06"`` for California.

    Returns
    -------
    List[Path]
        Paths to all downloaded files.
    """
    config = _load_config()

    if output_dir is None:
        output_dir = Path(config["data"]["raw_dir"]) / "human"
    output_dir = _ensure_dir(Path(output_dir))

    session = requests.Session()
    downloaded: List[Path] = []

    # ------------------------------------------------------------------
    # 1. Roads (TIGER/Line)
    # ------------------------------------------------------------------
    # California has 58 counties — download the primary roads file instead
    # for simplicity (full road network is large).
    roads_url = (
        f"https://www2.census.gov/geo/tiger/TIGER2023/PRISECROADS/"
        f"tl_2023_{state_fips}_prisecroads.zip"
    )
    roads_dest = output_dir / f"tl_2023_{state_fips}_prisecroads.zip"
    try:
        path = _stream_download(roads_url, roads_dest, session)
        downloaded.append(path)
    except requests.HTTPError as exc:
        logger.error("Roads download failed: %s", exc)

    # ------------------------------------------------------------------
    # 2. Powerlines (HIFLD GeoJSON export)
    # ------------------------------------------------------------------
    powerlines_url = (
        "https://opendata.arcgis.com/datasets/"
        "70592bc3d2894ea1880a7fb6c9f7c89b_0.geojson"
    )
    powerlines_dest = output_dir / "electric_power_transmission_lines.geojson"
    try:
        path = _stream_download(powerlines_url, powerlines_dest, session)
        downloaded.append(path)
    except requests.HTTPError as exc:
        logger.error("Powerlines download failed: %s", exc)

    # ------------------------------------------------------------------
    # 3. Population density (Census API)
    # ------------------------------------------------------------------
    census_key = os.environ.get("CENSUS_API_KEY", "")
    if not census_key:
        logger.warning(
            "CENSUS_API_KEY not set; skipping population density download. "
            "Get a free key at https://api.census.gov/data/key_signup.html"
        )
    else:
        census_url = (
            "https://api.census.gov/data/2022/acs/acs5"
            f"?get=NAME,B01003_001E&for=tract:*&in=state:{state_fips}&key={census_key}"
        )
        pop_dest = output_dir / "ca_population_by_tract.json"
        try:
            path = _stream_download(census_url, pop_dest, session)
            downloaded.append(path)
        except requests.HTTPError as exc:
            logger.error("Population download failed: %s", exc)

    logger.info("Human features: downloaded %d files to %s", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# gridMET (University of Idaho — daily 4 km surface meteorology)
# ---------------------------------------------------------------------------


def download_gridmet(
    start_year: int = 2015,
    end_year: int = 2023,
    variables: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Download gridMET daily 4 km surface meteorological NetCDF files for
    California's bounding box.

    **Data source**: University of Idaho Climatology Lab — gridMET
    https://www.climatologylab.org/gridmet.html

    gridMET is a high-resolution (~4 km) daily dataset covering the contiguous
    US from 1979 to the present.  Each file is one calendar year for one
    variable and covers the full CONUS.  California cells are extracted during
    the preprocessing stage.

    Files are served as NetCDF (.nc) from:
    ``http://www.northwestknowledge.net/metdata/data/{var}_{year}.nc``

    No authentication is required.

    Variables downloaded (configurable via config.yaml)
    ---------------------------------------------------
    tmmx  — Maximum near-surface air temperature (K)
    tmmn  — Minimum near-surface air temperature (K)
    pr    — Precipitation amount (mm)
    vs    — Wind speed at 10 m (m/s)
    rmax  — Maximum near-surface relative humidity (%)
    rmin  — Minimum near-surface relative humidity (%)
    srad  — Surface downwelling shortwave radiation (W/m²)
    vpd   — Mean vapor pressure deficit (kPa)

    Parameters
    ----------
    start_year:
        First year to download (inclusive).  Defaults to 2015.
    end_year:
        Last year to download (inclusive).  Defaults to 2023.
    variables:
        List of gridMET variable codes.  Defaults to all variables in config.
    output_dir:
        Destination directory.  Defaults to ``data/raw/gridmet/``.

    Returns
    -------
    List[Path]
        Paths to all downloaded NetCDF files.
    """
    config = _load_config()
    base_url = config["data"]["gridmet"]["base_url"]

    if variables is None:
        variables = list(config["data"]["gridmet"]["variables"].keys())
    if output_dir is None:
        output_dir = Path(config["data"]["raw_dir"]) / "gridmet"
    output_dir = _ensure_dir(Path(output_dir))

    session = requests.Session()
    downloaded: List[Path] = []

    for year in range(start_year, end_year + 1):
        year_dir = _ensure_dir(output_dir / str(year))
        for var in tqdm(variables, desc=f"gridMET {year}", leave=False):
            filename = f"{var}_{year}.nc"
            url = f"{base_url}{filename}"
            dest = year_dir / filename
            try:
                path = _stream_download(url, dest, session)
                downloaded.append(path)
            except requests.HTTPError as exc:
                logger.error("gridMET: failed to download %s (%s): %s", var, year, exc)
        # Brief pause between years to stay polite to the server
        time.sleep(0.5)

    logger.info("gridMET: downloaded %d files to %s", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# NDFD (NOAA National Digital Forecast Database — operational 7-day forecasts)
# ---------------------------------------------------------------------------


def download_ndfd(
    output_dir: Optional[Path] = None,
    elements: Optional[List[str]] = None,
    periods: Optional[List[str]] = None,
) -> List[Path]:
    """
    Download NOAA National Digital Forecast Database (NDFD) operational
    GRIB2 forecast files for the CONUS domain.

    **Data source**: NOAA National Weather Service — NDFD
    https://digital.weather.gov/
    File server: https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/

    NDFD provides gridded 2.5 km probabilistic and deterministic forecasts
    out to 7 days.  This function downloads the **current operational cycle**,
    which makes it suitable for real-time inference.  For historical model
    training (2015–2023), use gridMET or ERA5 as the weather source.

    File structure on the NOAA server
    ----------------------------------
    Each element is stored in period-specific subdirectories:
      VP.001-003/ds.{element}.bin   — days 1–3
      VP.004-007/ds.{element}.bin   — days 4–7

    Elements downloaded (configurable via config.yaml)
    --------------------------------------------------
    maxt  — Maximum temperature (K)
    mint  — Minimum temperature (K)
    wind  — Sustained wind speed (knots)
    rhm   — Relative humidity (%)
    qpf   — Quantitative precipitation forecast (in)
    sky   — Cloud cover (%)

    The GRIB2 files are decoded in the preprocessing stage using cfgrib/eccodes.

    Parameters
    ----------
    output_dir:
        Destination directory.  Defaults to ``data/raw/ndfd/``.
    elements:
        Override element code list from config (e.g. ``["maxt", "mint"]``).
    periods:
        Override period subdirectory list (e.g. ``["VP.001-003"]``).

    Returns
    -------
    List[Path]
        Paths to all downloaded GRIB2 files.

    Notes
    -----
    - Files are overwritten on each call since they represent the latest cycle.
    - Run this function daily (e.g. via a cron job) during the inference phase
      to keep the 7-day forecast window current.
    - Spatial subsetting to California is performed in preprocessing; the full
      CONUS GRIB2 file is downloaded here to avoid server-side clipping errors.
    """
    config = _load_config()
    base_url = config["data"]["ndfd"]["base_url"]

    if elements is None:
        elements = list(config["data"]["ndfd"]["elements"].keys())
    if periods is None:
        periods = config["data"]["ndfd"]["periods"]
    if output_dir is None:
        output_dir = Path(config["data"]["raw_dir"]) / "ndfd"
    output_dir = _ensure_dir(Path(output_dir))

    session = requests.Session()
    downloaded: List[Path] = []

    for period in periods:
        period_dir = _ensure_dir(output_dir / period)
        for element in tqdm(elements, desc=f"NDFD {period}", leave=False):
            filename = f"ds.{element}.bin"
            url = f"{base_url}{period}/{filename}"
            # NDFD operational files are replaced each cycle — always re-download
            dest = period_dir / filename
            if dest.exists():
                dest.unlink()

            try:
                logger.info("Downloading NDFD %s/%s", period, filename)
                response = session.get(url, stream=True, timeout=60)
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))
                with open(dest, "wb") as fh, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                    leave=False,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
                downloaded.append(dest)
            except requests.HTTPError as exc:
                logger.error("NDFD: failed to download %s/%s: %s", period, filename, exc)

        time.sleep(0.1)

    logger.info("NDFD: downloaded %d files to %s", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# ERA5 (ECMWF Reanalysis v5 — hourly single-level via Copernicus CDS API)
# ---------------------------------------------------------------------------


def download_era5(
    start_year: int = 2015,
    end_year: int = 2023,
    variables: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    bbox: Optional[dict] = None,
) -> List[Path]:
    """
    Download ERA5 hourly reanalysis data on single levels for California
    via the Copernicus Climate Data Store (CDS) API.

    **Data source**: ECMWF ERA5 Reanalysis — Copernicus CDS
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

    ERA5 provides hourly global reanalysis at ~31 km resolution from 1940 to
    present.  We download daily files (4 synoptic hours: 00, 06, 12, 18 UTC)
    for the California bounding box.  Daily aggregation (max, min, mean, sum)
    is performed in the preprocessing stage.

    Authentication
    --------------
    Requires a free Copernicus CDS account and API key.

    1. Register at https://cds.climate.copernicus.eu
    2. Go to your profile → copy your API key
    3. Create ``~/.cdsapirc``::

           url: https://cds.climate.copernicus.eu/api
           key: <your-api-key>

    Alternatively set environment variables ``CDSAPI_URL`` and ``CDSAPI_KEY``.

    Variables downloaded (configurable via config.yaml)
    ---------------------------------------------------
    2m_temperature                    — Near-surface air temperature (K)
    2m_dewpoint_temperature           — Dewpoint → relative humidity (K)
    10m_u_component_of_wind           — Eastward wind component (m/s)
    10m_v_component_of_wind           — Northward wind component (m/s)
    total_precipitation               — Accumulated precipitation (m)
    surface_solar_radiation_downwards — Downward solar radiation (J/m²)

    Parameters
    ----------
    start_year:
        First year to download (inclusive).
    end_year:
        Last year to download (inclusive).
    variables:
        CDS variable names to request.  Defaults to list in config.
    output_dir:
        Destination directory.  Defaults to ``data/raw/era5/``.
    bbox:
        Bounding box for spatial subsetting as
        ``{lat_max, lon_min, lat_min, lon_max}`` (CDS north/west/south/east
        convention).  Defaults to California bbox from config.

    Returns
    -------
    List[Path]
        Paths to all downloaded NetCDF files (one per year-month).

    Raises
    ------
    ImportError
        If the ``cdsapi`` package is not installed
        (``pip install cdsapi``).
    EnvironmentError
        If ``~/.cdsapirc`` is missing and environment variables are not set.
    """
    try:
        import cdsapi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "cdsapi is required for ERA5 downloads.  "
            "Install it with: pip install cdsapi"
        ) from exc

    config = _load_config()
    era5_cfg = config["data"]["era5"]

    if variables is None:
        variables = era5_cfg["variables"]
    if output_dir is None:
        output_dir = Path(config["data"]["raw_dir"]) / "era5"
    output_dir = _ensure_dir(Path(output_dir))

    if bbox is None:
        b = config["grid"]["bbox"]
        # CDS area format: [North, West, South, East]
        area = [b["lat_max"], b["lon_min"], b["lat_min"], b["lon_max"]]
    else:
        area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]

    # Honour environment-variable overrides so CI/CD can inject credentials
    cds_url = os.environ.get("CDSAPI_URL")
    cds_key = os.environ.get("CDSAPI_KEY")
    client_kwargs: dict = {}
    if cds_url and cds_key:
        client_kwargs = {"url": cds_url, "key": cds_key}
    elif not Path.home().joinpath(".cdsapirc").exists():
        raise EnvironmentError(
            "ERA5 download requires CDS credentials.\n"
            "  1. Register at: https://cds.climate.copernicus.eu\n"
            "  2. Go to your profile page and copy your API key.\n"
            "  3. Create ~/.cdsapirc with:\n"
            "       url: https://cds.climate.copernicus.eu/api\n"
            "       key: <your-api-key>\n"
            "  Or set env vars CDSAPI_URL + CDSAPI_KEY."
        )

    client = cdsapi.Client(**client_kwargs)
    downloaded: List[Path] = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            import calendar
            _, n_days = calendar.monthrange(year, month)
            days = [f"{d:02d}" for d in range(1, n_days + 1)]

            month_dir = _ensure_dir(output_dir / str(year))
            dest = month_dir / f"era5_{year}_{month:02d}.nc"

            if dest.exists():
                logger.info("ERA5: skipping %s — already downloaded.", dest.name)
                downloaded.append(dest)
                continue

            logger.info("ERA5: requesting %d-%02d (%d days, %d variables)…",
                        year, month, n_days, len(variables))

            request_params = {
                "product_type": era5_cfg["product_type"],
                "variable": variables,
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": era5_cfg["times"],
                "area": area,
                "format": era5_cfg["format"],
            }

            try:
                client.retrieve(era5_cfg["dataset"], request_params, str(dest))
                downloaded.append(dest)
                logger.info("ERA5: saved %s", dest)
            except Exception as exc:  # cdsapi raises generic Exception on API errors
                logger.error("ERA5: request failed for %d-%02d: %s", year, month, exc)

    logger.info("ERA5: downloaded %d files to %s", len(downloaded), output_dir)
    return downloaded


# ---------------------------------------------------------------------------
# Convenience: download everything
# ---------------------------------------------------------------------------


def download_all(
    start_year: int = 2015,
    end_year: int = 2023,
    skip_modis: bool = False,
    skip_gridmet: bool = False,
    skip_ndfd: bool = False,
    skip_era5: bool = False,
    skip_noaa: bool = False,
    skip_landfire: bool = False,
    skip_terrain: bool = False,
    skip_human: bool = False,
) -> dict:
    """
    Run all download functions in sequence and return a summary dict.

    Weather data hierarchy
    ----------------------
    - **gridMET** — Primary historical daily weather source (4 km, 1979-present).
      Used for lagged and static weather features in training.
    - **ERA5**    — High-quality reanalysis backup / validation source (31 km).
      Use when gridMET is unavailable or for broader climate context.
    - **NDFD**    — Operational 7-day forecast source.  Used as forecast features
      during real-time inference only (not historical training).
    - **NOAA CFSv2** — Legacy forecast source retained for backward compatibility.

    Parameters
    ----------
    start_year, end_year:
        Year range for time-series data (MODIS, gridMET, ERA5).
    skip_*:
        Set to True to bypass individual sources (useful for partial re-runs).

    Returns
    -------
    dict
        Mapping from source name to list of downloaded file paths.
    """
    results: dict = {}

    if not skip_modis:
        logger.info("=== Downloading MODIS fire data ===")
        results["modis"] = download_modis_fire(start_year, end_year)

    if not skip_gridmet:
        logger.info("=== Downloading gridMET daily weather (historical) ===")
        results["gridmet"] = download_gridmet(start_year, end_year)

    if not skip_ndfd:
        logger.info("=== Downloading NDFD operational 7-day forecasts ===")
        results["ndfd"] = download_ndfd()

    if not skip_era5:
        logger.info("=== Downloading ERA5 reanalysis ===")
        results["era5"] = download_era5(start_year, end_year)

    if not skip_noaa:
        logger.info("=== Downloading NOAA CFSv2 weather forecasts ===")
        results["noaa"] = download_noaa_weather(
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-12-31",
        )

    if not skip_landfire:
        logger.info("=== Downloading LANDFIRE vegetation/fuels ===")
        results["landfire"] = download_landfire()

    if not skip_terrain:
        logger.info("=== Downloading USGS terrain DEM ===")
        results["terrain"] = download_terrain()

    if not skip_human:
        logger.info("=== Downloading human infrastructure layers ===")
        results["human"] = download_human_features()

    return results
