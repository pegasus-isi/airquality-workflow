#!/usr/bin/env python3

"""
Fetch SAGE Continuum measurements for a single node (VSN) and write them out as
an air-quality catalog CSV.

This runs as a Pegasus job inside the workflow container, which is where
``sage_data_client`` is installed — the submit host does not need it.

Two modes:
  * live query (default) — hits the SAGE data API via ``sage_data_client``
  * ``--input`` — reads a pre-downloaded JSONL dump instead (no network)

The emitted CSV uses the same column names as the OpenAQ catalog so that
``extract_aqi_timeseries.py`` can consume either one unchanged.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# SAGE measurement name -> the pollutant parameter used by the AQI code in
# bin/extract_aqi_timeseries.py. Names not listed here are skipped unless
# --default-parameter is given.
DEFAULT_NAME_MAP = {
    "env.air_quality.conc": "pm25",
    "env.air_quality.pm2.5": "pm25",
    "env.pm25": "pm25",
    "env.air_quality.pm10.0": "pm10",
    "env.pm10": "pm10",
    "env.air_quality.o3": "o3",
    "env.air_quality.no2": "no2",
    "env.air_quality.so2": "so2",
    "env.air_quality.co": "co",
}

CATALOG_COLUMNS = [
    "location",
    "location_id",
    "parameter",
    "value",
    "unit",
    "datetime",
    "timestamp",
    "hour_bucket",
]


def split_list(values):
    """Accept both ``--names a b`` and the GUI's single ``--names a,b`` token."""
    out = []
    for value in values or []:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def map_name_to_parameter(name, name_map, default_parameter):
    """Resolve a SAGE measurement name to an AQI parameter, or None to skip."""
    if name in name_map:
        return name_map[name]
    return default_parameter or None


def write_catalog(rows, output_path):
    """Write ``rows`` to ``output_path``, always creating the file.

    An empty catalog still produces a header-only CSV so that Pegasus stage-out
    never fails on a missing file (a missing output puts the job on HOLD and
    hangs the DAG).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        pd.DataFrame(columns=CATALOG_COLUMNS).to_csv(output_path, index=False)
        return 0

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    if df.empty:
        pd.DataFrame(columns=CATALOG_COLUMNS).to_csv(output_path, index=False)
        return 0

    # Epoch seconds via subtraction rather than astype("int64"): the underlying
    # resolution of a tz-aware column is ns on pandas 1.x/2.x but us on 3.x, so
    # a fixed //10**9 divisor silently produces the wrong number.
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df["timestamp"] = (df["datetime"] - epoch).dt.total_seconds().astype("int64")
    df["hour_bucket"] = df["datetime"].dt.floor("h")
    df = df.sort_values("datetime")
    df.to_csv(output_path, index=False)
    return len(df)


def rows_from_jsonl(input_file, vsn, plugin, names, name_map, default_parameter):
    """Read measurements from a pre-downloaded SAGE JSONL dump."""
    rows = []
    skipped_names = set()

    with open(input_file, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = record.get("meta", {})
            if vsn and meta.get("vsn") != vsn:
                continue
            if plugin and meta.get("plugin") != plugin:
                continue

            name = record.get("name")
            if names and name not in names:
                continue

            parameter = map_name_to_parameter(name, name_map, default_parameter)
            if not parameter:
                skipped_names.add(name)
                continue

            location = vsn or meta.get("vsn") or meta.get("node") or "unknown"
            rows.append({
                "location": location,
                "location_id": location,
                "parameter": parameter,
                "value": record.get("value"),
                "unit": record.get("unit", "unknown"),
                "datetime": record.get("timestamp"),
            })

    if skipped_names:
        logging.warning(
            "Skipped unmapped measurement names: %s "
            "(use --default-parameter to map them)",
            ", ".join(sorted(n for n in skipped_names if n)),
        )
    return rows


def query_sage(start, end, vsn, plugin, names, retries, retry_delay):
    """Query the SAGE data API, retrying transient failures with backoff."""
    import sage_data_client

    filter_dict = {}
    if plugin:
        filter_dict["plugin"] = plugin
    if vsn:
        filter_dict["vsn"] = vsn
    # The API filter takes a single name; multi-name selection is applied
    # client-side below.
    if names and len(names) == 1:
        filter_dict["name"] = names[0]

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            logging.info(
                "Querying SAGE (attempt %d/%d): start=%s end=%s filter=%s",
                attempt, retries, start, end, filter_dict,
            )
            return sage_data_client.query(start=start, end=end, filter=filter_dict)
        except Exception as exc:  # network / API transients
            last_error = exc
            logging.warning("SAGE query failed (attempt %d/%d): %s",
                            attempt, retries, exc)
            if attempt < retries:
                delay = retry_delay * attempt
                logging.info("Retrying in %d s...", delay)
                time.sleep(delay)

    raise RuntimeError(f"SAGE query failed after {retries} attempts: {last_error}")


def rows_from_api(df, vsn, names, name_map, default_parameter):
    """Convert a ``sage_data_client`` DataFrame into catalog rows."""
    rows = []
    skipped_names = set()

    if df is None or df.empty:
        return rows

    if names and len(names) > 1:
        df = df[df["name"].isin(names)]

    for _, record in df.iterrows():
        name = record.get("name")
        parameter = map_name_to_parameter(name, name_map, default_parameter)
        if not parameter:
            skipped_names.add(name)
            continue

        location = record.get("meta.vsn") or record.get("meta.node") or vsn or "unknown"
        rows.append({
            "location": location,
            "location_id": location,
            "parameter": parameter,
            "value": record.get("value"),
            "unit": record.get("unit", "unknown"),
            "datetime": record.get("timestamp"),
        })

    if skipped_names:
        logging.warning(
            "Skipped unmapped measurement names: %s "
            "(use --default-parameter to map them)",
            ", ".join(sorted(n for n in skipped_names if n)),
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Fetch SAGE Continuum air-quality measurements for one node"
    )
    parser.add_argument("--vsn", required=True,
                        help="SAGE node VSN (for example W045)")
    parser.add_argument("--start", required=True,
                        help="Start time, RFC3339 (2026-01-14T00:00:00Z) or "
                             "a SAGE relative offset such as -24h")
    parser.add_argument("--end", required=True,
                        help="End time, RFC3339 or a SAGE relative offset")
    parser.add_argument("--plugin", default=None,
                        help="Filter by SAGE plugin image")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Measurement names, space- or comma-separated")
    parser.add_argument("--default-parameter", default=None,
                        choices=["pm25", "pm10", "o3", "no2", "so2", "co"],
                        help="Parameter to assign to measurement names that are "
                             "not in the built-in name map (default: skip them)")
    parser.add_argument("--input", default=None,
                        help="Read from a pre-downloaded SAGE JSONL dump "
                             "instead of querying the API")
    parser.add_argument("--output", required=True,
                        help="Output catalog CSV")
    parser.add_argument("--retries", type=int, default=3,
                        help="API query attempts before giving up (default: 3)")
    parser.add_argument("--retry-delay", type=int, default=15,
                        help="Base seconds between retries, scaled linearly "
                             "(default: 15)")

    args = parser.parse_args()
    names = split_list(args.names)

    try:
        if args.input:
            rows = rows_from_jsonl(
                args.input, args.vsn, args.plugin, names,
                DEFAULT_NAME_MAP, args.default_parameter,
            )
        else:
            df = query_sage(args.start, args.end, args.vsn, args.plugin, names,
                            args.retries, args.retry_delay)
            rows = rows_from_api(df, args.vsn, names,
                                 DEFAULT_NAME_MAP, args.default_parameter)
    except Exception as exc:
        # SAGE is the only source for this location, so this is fatal — but
        # still write the declared output first, otherwise Pegasus holds the
        # job on a stage-out error instead of reporting the real failure.
        logging.error("Failed to fetch SAGE data for %s: %s", args.vsn, exc)
        write_catalog([], args.output)
        return 1

    count = write_catalog(rows, args.output)
    if count == 0:
        logging.error(
            "No SAGE measurements for vsn=%s plugin=%s names=%s in %s..%s",
            args.vsn, args.plugin, names or "(any)", args.start, args.end,
        )
        return 1

    logging.info("Wrote %d measurements for %s to %s", count, args.vsn, args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(message)s", stream=sys.stdout, level=logging.INFO
    )
    sys.exit(main())
