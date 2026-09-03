#!/usr/bin/env python3

"""
Pegasus workflow generator for Air Quality Forecasting.

This extends the base air quality workflow with LSTM-based AQI forecasting capabilities.
It runs both the base pipeline (extraction, analysis, anomaly detection) and the
forecasting pipeline (historical data, feature prep, model training, prediction, visualization)
in parallel after data extraction.

Every argument has a default, so the workflow can be launched from the Pegasus
Studio GUI with nothing filled in: the defaults run SAGE node W045 over the last
full day. List arguments also accept the GUI's single comma-separated token
("a,b,c") in addition to the shell's space-separated form.

Usage:
    ./workflow_generator.py                                   # SAGE defaults
    ./workflow_generator.py --data-source openaq --location-ids 2178 \
                            --start-date 2024-01-15 \
                            --historical-days 90 \
                            --output workflow_forecast.yml
"""

import os
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime, timedelta

# Import Pegasus API
from Pegasus.api import *

# ---------------------------------------------------------------------------
# Defaults that make the GUI's "Run" usable without any input
# ---------------------------------------------------------------------------

# SAGE Continuum defaults — a node that publishes air-quality concentrations
# through the seanshahkarami/air-quality plugin.
DEFAULT_SAGE_VSN = ["W045"]
DEFAULT_SAGE_PLUGIN = "registry.sagecontinuum.org/seanshahkarami/air-quality:0.3.0"
DEFAULT_SAGE_NAMES = ["env.air_quality.conc"]

# OpenAQ defaults. 2178 is the location used throughout this repo's examples.
DEFAULT_OPENAQ_LOCATION_IDS = [2178]

# Named regions for OpenAQ. These are bounding boxes, not hard-coded location
# IDs: the IDs are resolved live against the OpenAQ v3 /locations endpoint at
# generation time, so they cannot go stale.
# Format: (min_lon, min_lat, max_lon, max_lat)
OPENAQ_REGIONS = {
    "los-angeles": (-118.668, 33.704, -118.155, 34.337),
    "bay-area":    (-122.610, 37.200, -121.700, 38.100),
    "new-york":    (-74.259, 40.477, -73.700, 40.918),
    "chicago":     (-87.940, 41.644, -87.524, 42.023),
    "houston":     (-95.789, 29.523, -95.014, 30.110),
    "denver":      (-105.110, 39.614, -104.600, 39.914),
    "seattle":     (-122.436, 47.491, -122.224, 47.734),
    "london":      (-0.510, 51.286, 0.334, 51.692),
    "delhi":       (76.840, 28.404, 77.348, 28.883),
    "beijing":     (116.000, 39.700, 116.800, 40.200),
}


def split_list(values, cast=str, flag=""):
    """Normalise a list argument.

    Accepts the shell form (``--names a b c``) and the Studio GUI form, which
    sends the whole list as one comma-separated token (``--names a,b,c``).
    """
    out = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(cast(part))
            except ValueError:
                raise ValueError(f"{flag}: cannot parse {part!r} as {cast.__name__}")
    return out


def parse_date(value, flag):
    """Parse a YYYY-MM-DD date, with an error message the GUI can surface."""
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{flag}: expected YYYY-MM-DD, got {value!r}")


def resolve_openaq_region(region, bbox, max_locations):
    """Resolve a named region or bbox to a list of OpenAQ location IDs."""
    if region:
        if region not in OPENAQ_REGIONS:
            raise ValueError(
                f"--openaq-region: unknown region {region!r}. "
                f"Known: {', '.join(sorted(OPENAQ_REGIONS))}"
            )
        bbox = OPENAQ_REGIONS[region]

    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from fetch_openaq_catalog import search_locations

    print(f"Resolving OpenAQ location IDs for bbox {bbox}...")
    df = search_locations(bbox=tuple(bbox))
    if df.empty:
        raise ValueError(
            f"No OpenAQ locations found in bbox {bbox}. "
            f"Pass --location-ids explicitly, or search with "
            f"./fetch_openaq_catalog.py --search --bbox {' '.join(str(b) for b in bbox)}"
        )

    ids = [int(i) for i in df["id"].head(max_locations).tolist()]
    for _, row in df.head(max_locations).iterrows():
        print(f"  {row['id']}: {row['name']}")
    return ids


class AirQualityForecastWorkflow:
    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "airquality_forecast"

    openaq_catalog = None
    openaq_cache_file = "openaq_catalog.csv"

    def __init__(
        self,
        location_ids,
        start_date,
        end_date,
        parameters,
        data_source="openaq",
        sage_input=None,
        sage_vsn=None,
        sage_plugin=None,
        sage_names=None,
        sage_default_parameter=None,
        historical_days=90,
        forecast_horizon=24,
        skip_forecast=False,
        dagfile="workflow_forecast.yml"
    ):
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")
        self.location_ids = location_ids or []
        self.parameters = parameters if parameters else ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = data_source
        self.sage_input = sage_input
        self.sage_vsn = sage_vsn or []
        self.sage_plugin = sage_plugin
        self.sage_names = sage_names
        self.sage_default_parameter = sage_default_parameter
        self.historical_days = historical_days
        self.forecast_horizon = forecast_horizon
        self.skip_forecast = skip_forecast
        self.historical_start_date = start_date - timedelta(days=historical_days)

    def write(self):
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    def create_pegasus_properties(self):
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"
        return

    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()

        local = Site("local").add_directories(
            Directory(
                Directory.SHARED_SCRATCH, self.shared_scratch_dir
            ).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(Directory.LOCAL_STORAGE, self.local_storage_dir).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
        )

        self.sc.add_sites(local, exec_site)

    def create_transformation_catalog(
        self,
        exec_site_name="condorpool",
        container_sif="Apptainer/AirQuality_Forecast_Container.sif",
    ):
        self.tc = TransformationCatalog()

        # Both containers below are backed by the same local Apptainer .sif
        # (the forecast image already carries the base stack plus PyTorch).
        # Pegasus stages the file like any other input, so image_site is the
        # site where the .sif physically lives (the submit host = "local").
        sif_path = (
            container_sif
            if os.path.isabs(container_sif)
            else os.path.join(self.wf_dir, container_sif)
        )
        if not os.path.exists(sif_path):
            print(f"Warning: Apptainer image not found at {sif_path} — build it "
                  f"first with: apptainer build {sif_path} "
                  f"Apptainer/AirQuality_Forecast_Container.def")
        image_url = "file://" + sif_path

        # Base workflow container
        airquality_container = Container(
            "airquality_container",
            container_type=Container.SINGULARITY,
            image=image_url,
            image_site="local",
        )

        # Forecast workflow container (with PyTorch)
        forecast_container = Container(
            "airquality_forecast_container",
            container_type=Container.SINGULARITY,
            image=image_url,
            image_site="local",
        )

        # Base transformations
        mkdir = Transformation(
            "mkdir", site="local", pfn="/bin/mkdir", is_stageable=False
        )

        # SAGE ingest runs as a job so that sage_data_client only has to exist
        # inside the container, never on the submit host.
        fetch_sage = Transformation(
            "fetch_sage",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/fetch_sage_data.py"),
            is_stageable=True,
            container=airquality_container,
        ).add_pegasus_profile(memory="2 GB")

        extract_timeseries = Transformation(
            "extract_timeseries",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/extract_aqi_timeseries.py"),
            is_stageable=True,
            container=airquality_container,
        ).add_pegasus_profile(memory="2 GB")

        analyze_pollutants = Transformation(
            "analyze_pollutants",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/analyze_pollutants.py"),
            is_stageable=True,
            container=airquality_container,
        ).add_pegasus_profile(memory="2 GB")

        detect_anomalies = Transformation(
            "detect_anomalies",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/detect_anomalies.py"),
            is_stageable=True,
            container=airquality_container,
        ).add_pegasus_profile(memory="1 GB")

        merge = Transformation(
            "merge",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/merge.py"),
            is_stageable=True,
            container=airquality_container,
        ).add_pegasus_profile(memory="1 GB")

        # Forecast transformations
        fetch_historical = Transformation(
            "fetch_historical",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/fetch_historical_data.py"),
            is_stageable=True,
            container=forecast_container,
        ).add_pegasus_profile(memory="2 GB")

        prepare_features = Transformation(
            "prepare_features",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/prepare_features.py"),
            is_stageable=True,
            container=forecast_container,
        ).add_pegasus_profile(memory="2 GB")

        train_model = Transformation(
            "train_model",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/train_forecast_model.py"),
            is_stageable=True,
            container=forecast_container,
        ).add_pegasus_profile(memory="4 GB")

        generate_forecast = Transformation(
            "generate_forecast",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/generate_forecast.py"),
            is_stageable=True,
            container=forecast_container,
        ).add_pegasus_profile(memory="2 GB")

        visualize_forecast = Transformation(
            "visualize_forecast",
            site=exec_site_name,
            pfn=os.path.join(self.wf_dir, "bin/visualize_forecast.py"),
            is_stageable=True,
            container=forecast_container,
        ).add_pegasus_profile(memory="2 GB")

        self.tc.add_containers(airquality_container, forecast_container)
        self.tc.add_transformations(
            mkdir, fetch_sage, extract_timeseries, analyze_pollutants, detect_anomalies, merge,
            fetch_historical, prepare_features, train_model, generate_forecast, visualize_forecast
        )

    def fetch_openaq_catalog(self):
        """Fetch air quality data from OpenAQ API v3."""
        print("Fetching OpenAQ data...")

        sys.path.insert(0, self.wf_dir)
        from fetch_openaq_catalog import fetch_openaq_catalog, save_catalog

        df = fetch_openaq_catalog(
            location_ids=self.location_ids,
            start_date=self.start_date,
            end_date=self.end_date,
            parameters=self.parameters
        )

        if df.empty:
            print("No data fetched from OpenAQ")
            return False

        save_catalog(df, self.openaq_cache_file)
        self.openaq_catalog = df
        return True

    def create_replica_catalog(self):
        """Register the workflow's generation-time inputs.

        SAGE data is fetched by the ``fetch_sage`` job at run time, so the only
        thing to register for that source is an optional pre-downloaded JSONL
        dump. OpenAQ is still fetched here, because the location *names* it
        returns determine the shape of the DAG.
        """
        self.rc = ReplicaCatalog()

        if self.data_source == "sage":
            if self.sage_input:
                input_path = Path(self.sage_input).resolve()
                if not input_path.exists():
                    print(f"Error: SAGE input file not found: {input_path}")
                    sys.exit(1)
                self.rc.add_replica("local", input_path.name, "file://" + str(input_path))
            return

        if self.openaq_catalog is None:
            if not self.fetch_openaq_catalog():
                print("Failed to fetch OpenAQ data")
                sys.exit(1)

        self.rc.add_replica(
            "local",
            "openaq_catalog.csv",
            "file://" + os.path.join(self.wf_dir, self.openaq_cache_file)
        )

    def create_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        # Get unique location names for each location ID (OpenAQ) or, for SAGE,
        # straight from the requested VSNs — the SAGE catalog does not exist yet
        # at generation time, it is produced by the fetch_sage jobs.
        location_map = {}
        if self.data_source == "sage":
            for vsn in self.sage_vsn:
                safe_name = vsn.replace(' ', '_').replace('-', '_').replace('/', '_')
                location_map[vsn] = {
                    "name": safe_name,
                    "display_name": vsn
                }
        else:
            if self.openaq_catalog is None or self.openaq_catalog.empty:
                print("Error: No catalog data available. Run fetch_openaq_catalog first.")
                return

            for loc_id in self.location_ids:
                loc_data = self.openaq_catalog[self.openaq_catalog['location_id'] == loc_id]
                if not loc_data.empty:
                    loc_name = loc_data['location'].iloc[0]
                    safe_name = loc_name.replace(' ', '_').replace('-', '_').replace('/', '_')
                    location_map[loc_id] = {
                        'name': safe_name,
                        'display_name': loc_name
                    }

        print(f"\nCreating workflow for {len(location_map)} location(s)")
        if not self.skip_forecast:
            print(f"Historical data period: {self.historical_days} days")
            print(f"Forecast horizon: {self.forecast_horizon} hours\n")
        else:
            print("Forecast pipeline: skipped\n")

        anomaly_files = []

        for loc_id, loc_info in location_map.items():
            location = loc_info['name']
            display_name = loc_info['display_name']

            if self.data_source == "sage":
                print(f"  Processing location: {display_name}")
            else:
                print(f"  Processing location: {display_name} (ID: {loc_id})")

            # Create directories
            mkdir_job = (
                Job(
                    "mkdir",
                    _id=f"mkdir_{location}",
                    node_label=f"mkdir_{location}",
                )
                .add_args(
                    f"-p {self.local_storage_dir}/timeseries/{location} "
                    f"{self.local_storage_dir}/analysis/{location} "
                    f"{self.local_storage_dir}/anomalies/{location} "
                    f"{self.local_storage_dir}/historical/{location} "
                    f"{self.local_storage_dir}/features/{location} "
                    f"{self.local_storage_dir}/models/{location} "
                    f"{self.local_storage_dir}/forecasts/{location}"
                )
                .add_profiles(
                    Namespace.SELECTOR, key="execution.site", value="local"
                )
            )
            self.wf.add_jobs(mkdir_job)

            # ===== INGEST =====

            if self.data_source == "sage":
                # One fetch job per node. sage_data_client lives in the
                # container, so nothing SAGE-specific is needed on the submit
                # host.
                catalog_file = File(f"catalog/{location}_catalog.csv")
                fetch_args = (
                    f"--vsn {display_name} "
                    f"--start {self.start_date.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                    f"--end {self.end_date.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                    f"--output catalog/{location}_catalog.csv"
                )
                if self.sage_plugin:
                    fetch_args += f" --plugin {self.sage_plugin}"
                if self.sage_names:
                    fetch_args += f" --names {' '.join(self.sage_names)}"
                if self.sage_default_parameter:
                    fetch_args += f" --default-parameter {self.sage_default_parameter}"

                fetch_sage_job = Job(
                    "fetch_sage",
                    _id=f"fetch_sage_{location}",
                    node_label=f"fetch_sage_{location}",
                )
                if self.sage_input:
                    sage_input_file = File(Path(self.sage_input).name)
                    fetch_args += f" --input {sage_input_file.lfn}"
                    fetch_sage_job.add_inputs(sage_input_file)

                (
                    fetch_sage_job
                    .add_args(fetch_args)
                    .add_outputs(catalog_file, stage_out=False, register_replica=False)
                    .add_dagman_profile(retry="2")
                    .add_pegasus_profiles(label=location)
                )
                self.wf.add_jobs(fetch_sage_job)
                self.wf.add_dependency(mkdir_job, children=[fetch_sage_job])
            else:
                # OpenAQ is fetched at generation time — the location names it
                # returns are what the DAG is built around.
                catalog_file = File("openaq_catalog.csv")
                fetch_sage_job = None

            # ===== BASE PIPELINE =====

            # Extract time series (shared by both pipelines)
            timeseries_file = File(f"timeseries/{location}/{location}_timeseries.json")
            extract_job = (
                Job(
                    "extract_timeseries",
                    _id=f"extract_{location}",
                    node_label=f"extract_{location}",
                )
                .add_args(f"-i {catalog_file.lfn} -o timeseries/{location}")
                .add_inputs(catalog_file)
                .add_outputs(timeseries_file, stage_out=False, register_replica=False)
                .add_pegasus_profiles(label=location)
            )
            self.wf.add_jobs(extract_job)
            self.wf.add_dependency(mkdir_job, children=[extract_job])
            if fetch_sage_job is not None:
                self.wf.add_dependency(fetch_sage_job, children=[extract_job])

            # Analyze pollutants
            analysis_png = File(f"analysis/{location}/{location}_analysis.png")
            stats_file = File(f"analysis/{location}/{location}_statistics.json")
            analyze_job = (
                Job(
                    "analyze_pollutants",
                    _id=f"analyze_{location}",
                    node_label=f"analyze_{location}",
                )
                .add_args(f"-i timeseries/{location}/{location}_timeseries.json -o analysis/{location}")
                .add_inputs(timeseries_file)
                .add_outputs(
                    analysis_png, stats_file,
                    stage_out=True, register_replica=False
                )
                .add_pegasus_profiles(label=location)
            )
            self.wf.add_jobs(analyze_job)

            # Detect anomalies
            anomaly_file = File(f"anomalies/{location}/{location}_anomalies.json")
            anomaly_files.append(anomaly_file)
            anomaly_job = (
                Job(
                    "detect_anomalies",
                    _id=f"anomaly_{location}",
                    node_label=f"anomaly_{location}",
                )
                .add_args(
                    f"-i timeseries/{location}/{location}_timeseries.json "
                    f"-o anomalies/{location}/{location}_anomalies.json -t 3.0"
                )
                .add_inputs(timeseries_file)
                .add_outputs(anomaly_file, stage_out=True, register_replica=False)
                .add_pegasus_profiles(label=location)
            )
            self.wf.add_jobs(anomaly_job)

            # ===== FORECAST PIPELINE =====
            if self.skip_forecast:
                continue

            # Fetch historical data (90 days)
            historical_file = File(f"historical/{location}/{location}_historical.csv")
            fetch_hist_job = (
                Job(
                    "fetch_historical",
                    _id=f"fetch_hist_{location}",
                    node_label=f"fetch_hist_{location}",
                )
                .add_args(
                    f"--location-id {loc_id} "
                    f"--days {self.historical_days} "
                    f"--end-date {self.start_date.strftime('%Y-%m-%d')} "
                    f"--output historical/{location}/{location}_historical.csv"
                )
                .add_outputs(historical_file, stage_out=False, register_replica=False)
                .add_env(OPENAQ_API_KEY=os.environ.get('OPENAQ_API_KEY', ''))
                .add_pegasus_profiles(label=f"{location}_forecast")
            )
            self.wf.add_jobs(fetch_hist_job)
            self.wf.add_dependency(mkdir_job, children=[fetch_hist_job])

            # Prepare features (depends on both timeseries and historical data)
            features_file = File(f"features/{location}/{location}_train.npz")
            scaler_file = File(f"features/{location}/{location}_train_scaler.json")
            prepare_job = (
                Job(
                    "prepare_features",
                    _id=f"prepare_{location}",
                    node_label=f"prepare_{location}",
                )
                .add_args(
                    f"--timeseries timeseries/{location}/{location}_timeseries.json "
                    f"--historical historical/{location}/{location}_historical.csv "
                    f"--output features/{location}/{location}_train.npz "
                    f"--lookback 168 "
                    f"--horizon {self.forecast_horizon}"
                )
                .add_inputs(timeseries_file, historical_file)
                .add_outputs(
                    features_file, scaler_file,
                    stage_out=False, register_replica=False
                )
                .add_pegasus_profiles(label=f"{location}_forecast")
            )
            self.wf.add_jobs(prepare_job)
            # Depends on both extract and fetch_hist
            self.wf.add_dependency(extract_job, children=[prepare_job])
            self.wf.add_dependency(fetch_hist_job, children=[prepare_job])

            # Train LSTM model
            model_checkpoint = File(f"models/{location}/{location}_lstm_checkpoint.pt")
            training_info = File(f"models/{location}/{location}_training_info.json")
            train_job = (
                Job(
                    "train_model",
                    _id=f"train_{location}",
                    node_label=f"train_{location}",
                )
                .add_args(
                    f"--features features/{location}/{location}_train.npz "
                    f"--output models/{location} "
                    f"--location-name {location} "
                    f"--epochs 100 "
                    f"--batch-size 32 "
                    f"--patience 10"
                )
                .add_inputs(features_file, scaler_file)
                .add_outputs(
                    model_checkpoint, training_info,
                    stage_out=True, register_replica=False
                )
                .add_pegasus_profiles(label=f"{location}_forecast")
            )
            self.wf.add_jobs(train_job)
            self.wf.add_dependency(prepare_job, children=[train_job])

            # Generate forecast
            forecast_file = File(f"forecasts/{location}/{location}_forecast.json")
            forecast_job = (
                Job(
                    "generate_forecast",
                    _id=f"forecast_{location}",
                    node_label=f"forecast_{location}",
                )
                .add_args(
                    f"--model models/{location}/{location}_lstm_checkpoint.pt "
                    f"--timeseries timeseries/{location}/{location}_timeseries.json "
                    f"--scaler features/{location}/{location}_train_scaler.json "
                    f"--output forecasts/{location}/{location}_forecast.json "
                    f"--location-name \"{display_name}\" "
                    f"--lookback 168"
                )
                .add_inputs(model_checkpoint, timeseries_file, scaler_file)
                .add_outputs(forecast_file, stage_out=True, register_replica=False)
                .add_pegasus_profiles(label=f"{location}_forecast")
            )
            self.wf.add_jobs(forecast_job)
            self.wf.add_dependency(train_job, children=[forecast_job])

            # Visualize forecast
            forecast_viz = File(f"forecasts/{location}/{location}_forecast.png")
            forecast_summary = File(f"forecasts/{location}/{location}_forecast_summary.json")
            viz_job = (
                Job(
                    "visualize_forecast",
                    _id=f"viz_forecast_{location}",
                    node_label=f"viz_forecast_{location}",
                )
                .add_args(
                    f"--timeseries timeseries/{location}/{location}_timeseries.json "
                    f"--forecast forecasts/{location}/{location}_forecast.json "
                    f"--output forecasts/{location}/{location}_forecast.png "
                    f"--lookback-days 7"
                )
                .add_inputs(timeseries_file, forecast_file)
                .add_outputs(
                    forecast_viz, forecast_summary,
                    stage_out=True, register_replica=False
                )
                .add_pegasus_profiles(label=f"{location}_forecast")
            )
            self.wf.add_jobs(viz_job)
            self.wf.add_dependency(forecast_job, children=[viz_job])

        # Merge all anomaly results (base workflow final step)
        if len(anomaly_files) > 1:
            merged_anomalies = File("merged_anomalies.json")
            merge_job = (
                Job(
                    "merge",
                    _id="merge_all_anomalies",
                    node_label="merge_all",
                )
                .add_args(
                    f"-i {' '.join([f.lfn for f in anomaly_files])} -o {merged_anomalies.lfn}"
                )
                .add_inputs(*anomaly_files)
                .add_outputs(merged_anomalies, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(merge_job)

        print("\nWorkflow created successfully!")
        print(f"  Base pipeline: extract → analyze → anomaly detection")
        if not self.skip_forecast:
            print("  Forecast pipeline: fetch historical → prepare features → train LSTM → forecast → visualize")


if __name__ == "__main__":
    parser = ArgumentParser(description="Pegasus Air Quality Forecast Workflow")

    parser.add_argument(
        "-s",
        "--skip-sites-catalog",
        action="store_true",
        help="Skip site catalog creation",
    )
    parser.add_argument(
        "-e",
        "--execution-site-name",
        metavar="STR",
        type=str,
        default="condorpool",
        help="Execution site name (default: condorpool)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="STR",
        type=str,
        default="workflow_forecast.yml",
        help="Output file (default: workflow_forecast.yml)",
    )
    parser.add_argument(
        "--location-ids",
        metavar="INT",
        type=str,
        nargs="+",
        default=[str(i) for i in DEFAULT_OPENAQ_LOCATION_IDS],
        help="OpenAQ location IDs, space- or comma-separated (default: "
             f"{','.join(str(i) for i in DEFAULT_OPENAQ_LOCATION_IDS)}). "
             "Ignored when --openaq-region or --openaq-bbox is given. "
             "Find more with ./fetch_openaq_catalog.py --search",
    )
    parser.add_argument(
        "--openaq-region",
        choices=sorted(OPENAQ_REGIONS),
        default=None,
        help="Pick OpenAQ locations by named region instead of by ID; the IDs "
             "are resolved live against the OpenAQ API",
    )
    parser.add_argument(
        "--openaq-bbox",
        metavar="FLOAT",
        type=str,
        nargs="+",
        default=None,
        help="Pick OpenAQ locations inside an arbitrary bounding box: "
             "min_lon min_lat max_lon max_lat (space- or comma-separated)",
    )
    parser.add_argument(
        "--openaq-max-locations",
        metavar="INT",
        type=int,
        default=3,
        help="How many locations to take from a region/bbox search (default: 3)",
    )
    parser.add_argument(
        "--start-date",
        metavar="STR",
        type=str,
        default=None,
        help="Start date, YYYY-MM-DD (default: yesterday, UTC)",
    )
    parser.add_argument(
        "--end-date",
        metavar="STR",
        type=str,
        default=None,
        help="End date, YYYY-MM-DD (default: start date + 1 day)",
    )
    parser.add_argument(
        "--parameters",
        metavar="STR",
        type=str,
        nargs="+",
        default=None,
        help="Parameters to analyze, space- or comma-separated, from "
             "pm25 pm10 o3 no2 so2 co (default: all)",
    )
    parser.add_argument(
        "--historical-days",
        metavar="INT",
        type=int,
        default=90,
        help="Days of historical data for training (default: 90)",
    )
    parser.add_argument(
        "--forecast-horizon",
        metavar="INT",
        type=int,
        default=24,
        help="Forecast horizon in hours (default: 24)",
    )
    parser.add_argument(
        "--data-source",
        choices=["sage", "openaq"],
        default="sage",
        help="Data source (default: sage). SAGE needs no API key and is "
             "fetched by an in-container job; openaq needs OPENAQ_API_KEY set "
             "on the submit host",
    )
    parser.add_argument(
        "--sage-input",
        type=str,
        default=None,
        help="Optional pre-downloaded SAGE JSONL dump; staged in and read by "
             "the fetch_sage job instead of querying the SAGE API",
    )
    parser.add_argument(
        "--sage-vsn",
        type=str,
        nargs="+",
        default=list(DEFAULT_SAGE_VSN),
        help="SAGE node VSNs, space- or comma-separated. One fetch job and one "
             f"pipeline per node (default: {','.join(DEFAULT_SAGE_VSN)})",
    )
    parser.add_argument(
        "--sage-plugin",
        type=str,
        default=DEFAULT_SAGE_PLUGIN,
        help=f"Filter SAGE data by plugin (default: {DEFAULT_SAGE_PLUGIN})",
    )
    parser.add_argument(
        "--sage-names",
        type=str,
        nargs="+",
        default=list(DEFAULT_SAGE_NAMES),
        help="SAGE measurement names, space- or comma-separated "
             f"(default: {','.join(DEFAULT_SAGE_NAMES)})",
    )
    parser.add_argument(
        "--sage-default-parameter",
        choices=['pm25', 'pm10', 'o3', 'no2', 'so2', 'co'],
        default=None,
        help="Pollutant to assign to SAGE measurement names that the fetch job "
             "does not recognise (default: skip unrecognised names)",
    )
    parser.add_argument(
        "--container-sif",
        type=str,
        default="Apptainer/AirQuality_Forecast_Container.sif",
        help="Path to the Apptainer .sif image, absolute or relative to the "
             "workflow directory (default: Apptainer/AirQuality_Forecast_Container.sif)",
    )
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip LSTM forecast pipeline",
    )

    args = parser.parse_args()

    try:
        # --- normalise list arguments (accept the GUI's comma form) ---
        args.sage_vsn = split_list(args.sage_vsn, str, "--sage-vsn")
        args.sage_names = split_list(args.sage_names, str, "--sage-names")
        args.location_ids = split_list(args.location_ids, int, "--location-ids")
        if args.parameters:
            args.parameters = split_list(args.parameters, str, "--parameters")
            unknown = set(args.parameters) - {'pm25', 'pm10', 'o3', 'no2', 'so2', 'co'}
            if unknown:
                raise ValueError(
                    f"--parameters: unknown parameter(s) {', '.join(sorted(unknown))}"
                )
        if args.openaq_bbox:
            args.openaq_bbox = split_list(args.openaq_bbox, float, "--openaq-bbox")
            if len(args.openaq_bbox) != 4:
                raise ValueError(
                    "--openaq-bbox: expected 4 values (min_lon min_lat max_lon max_lat)"
                )

        # --- dates: default to the last full UTC day ---
        if args.start_date:
            args.start_date = parse_date(args.start_date, "--start-date")
        else:
            today = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            args.start_date = today - timedelta(days=1)
            print(f"No --start-date given, defaulting to {args.start_date.date()}")

        if args.end_date:
            args.end_date = parse_date(args.end_date, "--end-date")
        else:
            args.end_date = args.start_date + timedelta(days=1)

        if args.end_date <= args.start_date:
            raise ValueError("--end-date must be after --start-date")

        # --- source-specific validation ---
        if args.data_source == "sage":
            if not args.sage_vsn:
                raise ValueError("--sage-vsn: at least one node VSN is required")
            if not args.skip_forecast:
                print("Note: SAGE has no OpenAQ history endpoint — "
                      "skipping the LSTM forecast pipeline.")
                args.skip_forecast = True
        else:
            if args.openaq_region or args.openaq_bbox:
                args.location_ids = resolve_openaq_region(
                    args.openaq_region, args.openaq_bbox, args.openaq_max_locations
                )
            if not args.location_ids:
                raise ValueError(
                    "--location-ids is required when --data-source is openaq "
                    "(or use --openaq-region / --openaq-bbox)"
                )
            if not os.environ.get("OPENAQ_API_KEY"):
                raise ValueError(
                    "OPENAQ_API_KEY is not set. Export it before generating an "
                    "OpenAQ workflow, or use --data-source sage (the default), "
                    "which needs no key."
                )

        print("=" * 70)
        print("AIR QUALITY FORECAST WORKFLOW GENERATOR")
        print("=" * 70)
        print(f"Data source: {args.data_source}")
        if args.data_source == "openaq":
            print(f"Location IDs: {args.location_ids}")
        else:
            print(f"SAGE nodes: {args.sage_vsn}")
            print(f"SAGE plugin: {args.sage_plugin}")
            print(f"SAGE names: {args.sage_names}")
            print(f"SAGE input: {args.sage_input or '(live API query)'}")
        print(f"Analysis period: {args.start_date.date()} to {args.end_date.date()}")
        print(f"Historical training data: {args.historical_days} days")
        print(f"Forecast horizon: {args.forecast_horizon} hours")
        print(f"Execution site: {args.execution_site_name}")
        print("=" * 70)

        workflow = AirQualityForecastWorkflow(
            location_ids=args.location_ids,
            start_date=args.start_date,
            end_date=args.end_date,
            parameters=args.parameters,
            data_source=args.data_source,
            sage_input=args.sage_input,
            sage_vsn=args.sage_vsn,
            sage_plugin=args.sage_plugin,
            sage_names=args.sage_names,
            sage_default_parameter=args.sage_default_parameter,
            historical_days=args.historical_days,
            forecast_horizon=args.forecast_horizon,
            skip_forecast=args.skip_forecast,
            dagfile=args.output
        )

        print("\nGenerating workflow...")
        workflow.create_pegasus_properties()

        if not args.skip_sites_catalog:
            workflow.create_sites_catalog(exec_site_name=args.execution_site_name)

        workflow.create_transformation_catalog(
            exec_site_name=args.execution_site_name,
            container_sif=args.container_sif,
        )
        workflow.create_replica_catalog()
        workflow.create_workflow()
        workflow.write()

        print(f"\n✓ Workflow written to {args.output}")
        print(f"\nTo submit the workflow:")
        print(f"  pegasus-plan --submit -s {args.execution_site_name} -o local {args.output}")

    except Exception as e:
        print(f"\n✗ Error creating workflow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
