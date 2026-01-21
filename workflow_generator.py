#!/usr/bin/env python3

"""
Pegasus workflow generator for Air Quality Forecasting.

This extends the base air quality workflow with LSTM-based AQI forecasting capabilities.
It runs both the base pipeline (extraction, analysis, anomaly detection) and the
forecasting pipeline (historical data, feature prep, model training, prediction, visualization)
in parallel after data extraction.

Usage:
    ./workflow_generator_forecast.py --location-ids 2178 \
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
        self.sage_vsn = sage_vsn
        self.sage_plugin = sage_plugin
        self.sage_names = sage_names
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

    def create_transformation_catalog(self, exec_site_name="condorpool"):
        self.tc = TransformationCatalog()

        # Base workflow container
        airquality_container = Container(
            "airquality_container",
            container_type=Container.SINGULARITY,
            image="docker://kthare10/airquality-forecast:latest",
            image_site="docker_hub",
        )

        # Forecast workflow container (with PyTorch)
        forecast_container = Container(
            "airquality_forecast_container",
            container_type=Container.SINGULARITY,
            image="docker://kthare10/airquality-forecast:latest",
            image_site="docker_hub",
        )

        # Base transformations
        mkdir = Transformation(
            "mkdir", site="local", pfn="/bin/mkdir", is_stageable=False
        )

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
            mkdir, extract_timeseries, analyze_pollutants, detect_anomalies, merge,
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

    def load_sage_catalog(self):
        """Load SAGE data via JSONL file or sage_data_client and convert to catalog CSV."""
        def map_name_to_parameter(name: str):
            if name in ("env.air_quality.conc", "env.pm25"):
                return "pm25"
            if name in ("env.pm10",):
                return "pm10"
            return None

        rows = []

        if self.sage_input:
            input_path = Path(self.sage_input)
            if not input_path.exists():
                print(f"Error: SAGE input file not found: {input_path}")
                return False

            import json
            with open(input_path, "r") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    meta = record.get("meta", {})
                    if self.sage_vsn and meta.get("vsn") != self.sage_vsn:
                        continue
                    if self.sage_plugin and meta.get("plugin") != self.sage_plugin:
                        continue

                    name = record.get("name")
                    if self.sage_names and name not in self.sage_names:
                        continue

                    parameter = map_name_to_parameter(name)
                    if not parameter:
                        continue

                    location = meta.get("vsn") or meta.get("node") or "unknown"
                    rows.append({
                        "location": location,
                        "location_id": location,
                        "parameter": parameter,
                        "value": record.get("value"),
                        "unit": record.get("unit", "unknown"),
                        "datetime": record.get("timestamp"),
                    })
        else:
            try:
                import sage_data_client
            except ImportError:
                print("Error: sage_data_client is not available. Install it or use --sage-input.")
                return False

            start = self.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            end = self.end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            filter_dict = {}
            if self.sage_plugin:
                filter_dict["plugin"] = self.sage_plugin
            if self.sage_vsn:
                filter_dict["vsn"] = self.sage_vsn
            if self.sage_names and len(self.sage_names) == 1:
                filter_dict["name"] = self.sage_names[0]

            df = sage_data_client.query(start=start, end=end, filter=filter_dict)
            if df is None or df.empty:
                print("No SAGE measurements returned for the specified filters.")
                return False

            if self.sage_names and len(self.sage_names) > 1:
                df = df[df["name"].isin(self.sage_names)]

            for _, record in df.iterrows():
                name = record.get("name")
                parameter = map_name_to_parameter(name)
                if not parameter:
                    continue

                location = record.get("meta.vsn") or record.get("meta.node") or "unknown"
                rows.append({
                    "location": location,
                    "location_id": location,
                    "parameter": parameter,
                    "value": record.get("value"),
                    "unit": record.get("unit", "unknown"),
                    "datetime": record.get("timestamp"),
                })

        if not rows:
            print("No SAGE measurements matched the provided filters.")
            return False

        import pandas as pd
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.dropna(subset=["datetime"])
        if df.empty:
            print("No valid SAGE measurements after date parsing.")
            return False

        df["timestamp"] = df["datetime"].astype(int) // 10**9
        df["hour_bucket"] = df["datetime"].dt.floor("h")

        output_path = os.path.join(self.wf_dir, self.openaq_cache_file)
        df.to_csv(output_path, index=False)
        self.openaq_catalog = df
        return True

    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()

        if self.openaq_catalog is None:
            if self.data_source == "sage":
                if not self.load_sage_catalog():
                    print("Failed to load SAGE data")
                    sys.exit(1)
            else:
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

        catalog_file = File("openaq_catalog.csv")

        if self.openaq_catalog is None or self.openaq_catalog.empty:
            print("Error: No catalog data available. Run fetch_openaq_catalog first.")
            return

        # Get unique location names for each location ID (OpenAQ) or from SAGE data
        location_map = {}
        if self.data_source == "sage":
            for loc_name in sorted(self.openaq_catalog["location"].unique()):
                safe_name = loc_name.replace(' ', '_').replace('-', '_').replace('/', '_')
                location_map[loc_name] = {
                    "name": safe_name,
                    "display_name": loc_name
                }
        else:
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

            # ===== BASE PIPELINE =====

            # Extract time series (shared by both pipelines)
            timeseries_file = File(f"timeseries/{location}/{location}_timeseries.json")
            extract_job = (
                Job(
                    "extract_timeseries",
                    _id=f"extract_{location}",
                    node_label=f"extract_{location}",
                )
                .add_args(f"-i openaq_catalog.csv -o timeseries/{location}")
                .add_inputs(catalog_file)
                .add_outputs(timeseries_file, stage_out=False, register_replica=False)
                .add_pegasus_profiles(label=location)
            )
            self.wf.add_jobs(extract_job)
            self.wf.add_dependency(mkdir_job, children=[extract_job])

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
        type=int,
        required=False,
        nargs="+",
        help="OpenAQ location IDs (use fetch_openaq_catalog.py --search to find IDs)",
    )
    parser.add_argument(
        "--start-date",
        metavar="STR",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help="Start date (example: '2024-01-15')",
    )
    parser.add_argument(
        "--end-date",
        metavar="STR",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=None,
        help="End date (default: Start date + 1 day)",
    )
    parser.add_argument(
        "--parameters",
        metavar="STR",
        type=str,
        nargs="+",
        choices=['pm25', 'pm10', 'o3', 'no2', 'so2', 'co'],
        default=None,
        help="Parameters to analyze (default: all)",
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
        choices=["openaq", "sage"],
        default="openaq",
        help="Data source (default: openaq)",
    )
    parser.add_argument(
        "--sage-input",
        type=str,
        default=None,
        help="Path to SAGE JSONL data file (required when data-source is sage)",
    )
    parser.add_argument(
        "--sage-vsn",
        type=str,
        default=None,
        help="Filter SAGE data by VSN (optional)",
    )
    parser.add_argument(
        "--sage-plugin",
        type=str,
        default=None,
        help="Filter SAGE data by plugin (optional)",
    )
    parser.add_argument(
        "--sage-names",
        type=str,
        nargs="+",
        default=None,
        help="Filter SAGE data by measurement names (optional)",
    )
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip LSTM forecast pipeline",
    )

    args = parser.parse_args()

    if not args.end_date:
        args.end_date = args.start_date + timedelta(days=1)

    print("=" * 70)
    print("AIR QUALITY FORECAST WORKFLOW GENERATOR")
    print("=" * 70)
    print(f"Data source: {args.data_source}")
    if args.data_source == "openaq":
        print(f"Location IDs: {args.location_ids}")
    else:
        print(f"SAGE input: {args.sage_input}")
    print(f"Analysis period: {args.start_date.date()} to {args.end_date.date()}")
    print(f"Historical training data: {args.historical_days} days")
    print(f"Forecast horizon: {args.forecast_horizon} hours")
    print(f"Execution site: {args.execution_site_name}")
    print("=" * 70)

    try:
        if args.data_source == "openaq" and not args.location_ids:
            raise ValueError("--location-ids is required when data-source is openaq")
        if args.data_source == "sage" and not args.sage_input:
            try:
                import sage_data_client  # noqa: F401
            except ImportError:
                raise ValueError("--sage-input is required when data-source is sage unless sage_data_client is installed")
        if args.data_source == "sage" and not args.skip_forecast:
            print("Warning: SAGE data does not include OpenAQ history. Skipping forecast pipeline.")
            args.skip_forecast = True

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
            historical_days=args.historical_days,
            forecast_horizon=args.forecast_horizon,
            skip_forecast=args.skip_forecast,
            dagfile=args.output
        )

        print("\nGenerating workflow...")
        workflow.create_pegasus_properties()

        if not args.skip_sites_catalog:
            workflow.create_sites_catalog(exec_site_name=args.execution_site_name)

        workflow.create_transformation_catalog(exec_site_name=args.execution_site_name)
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
